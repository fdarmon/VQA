--------------------------------------------------------------------------------
-- Script adapted from eval.lua in https://github.com/GT-Vision-Lab/VQA_LSTM_CNN
--------------------------------------------------------------------------------

require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim'
require 'hdf5'
LSTM=require 'misc.LSTM'
cjson=require('cjson');
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('LSTM features extraction for both train and test set')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_ques_h5','data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-model_path', 'lstm.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-out_path', 'result/', 'path to save output h5 file')
cmd:option('-dataset','test','train or test (default train)')

-- batch size parameter to fit into gpu memory
cmd:option('-batch_size',500,'batch_size for each iterations')

-- Model parameter settings (shoud be the same with the training)
-- Leave  default parameter for using github's pretrained  model
cmd:option('-input_encoding_size', 200, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',2,'number of the rnn layer')

cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

opt = cmd:parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.setDevice(opt.gpuid + 1)
end


------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------
local model_path = opt.model_path
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local lstm_size_q=opt.rnn_size
local nlstm_layers_q=opt.rnn_layer
local nhimage=4096
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
print('DataLoader loading h5 file: ', opt.input_json)
local dataset = {}

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')

if opt.dataset =='test' then
    dataset['question'] = h5_file:read('/ques_test'):all()
    dataset['lengths_q'] = h5_file:read('/ques_length_test'):all()
    dataset['ques_id'] = h5_file:read('/question_id_test'):all()
    h5_file:close()
else
    dataset['question'] = h5_file:read('/ques_train'):all()
    dataset['lengths_q'] = h5_file:read('/ques_length_train'):all()
    dataset['ques_id'] = h5_file:read('/question_id_train'):all()
    h5_file:close()
end
dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count
collectgarbage();

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
buffer_size_q=dataset['question']:size()[2]


--embedding: word-embedding
embedding_net_q=nn.Sequential()
				:add(nn.Linear(vocabulary_size_q,embedding_size_q))
				:add(nn.Dropout(0.5))
				:add(nn.Tanh())
--encoder: RNN body
encoder_net_q=LSTM.lstm_conventional(embedding_size_q,lstm_size_q,dummy_output_size,nlstm_layers_q,0.5)

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)

if opt.gpuid >= 0 then
	print('shipped data function to cuda...')
	embedding_net_q = embedding_net_q:cuda()
	encoder_net_q = encoder_net_q:cuda()

	dummy_state_q = dummy_state_q:cuda()
	dummy_output_q = dummy_output_q:cuda()
end

-- setting to evaluation
embedding_net_q:evaluate();
encoder_net_q:evaluate();


embedding_w_q,embedding_dw_q=embedding_net_q:getParameters();
encoder_w_q,encoder_dw_q=encoder_net_q:getParameters();

-- loading the model
model_param=torch.load(model_path);
embedding_w_q:copy(model_param['embedding_w_q']);
encoder_w_q:copy(model_param['encoder_w_q']);


sizes={encoder_w_q:size(1),embedding_w_q:size(1)};

------------------------------------------------------------------------
--Grab Next Batch--
------------------------------------------------------------------------
function dataset:next_batch_test(s,e)
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=s+i-1;
	end

	local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question']:index(1,qinds),dataset['lengths_q']:index(1,qinds),vocabulary_size_q);

	local qids=dataset['ques_id']:index(1,qinds);

	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q[1]=fv_sorted_q[1]:cuda() --one hot encoding
		fv_sorted_q[3]=fv_sorted_q[3]:cuda() -- sorting index
		fv_sorted_q[4]=fv_sorted_q[4]:cuda() -- inverse  sorting  index
	end

	--print(string.format('batch_sort:%f',timer:time().real));
	return fv_sorted_q,qids,batch_size;
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------
-- duplicate the RNN
local encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q);
function forward(s,e)
	local timer = torch.Timer();
	--grab a batch--
	local fv_sorted_q,qids,batch_size=dataset:next_batch_test(s,e);
	local question_max_length=fv_sorted_q[2]:size(1);

	--embedding forward--
	local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]);

	--encoder forward--
	local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]);

	--multimodal/criterion forward--
	local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4]);

	return tv_q:double(),qids;
end


-----------------------------------------------------------------------
-- Pass through the whole dataset
-----------------------------------------------------------------------
nqs=dataset['question']:size(1);
fv=torch.Tensor(nqs,2*nlstm_layers_q*lstm_size_q);
qids=torch.LongTensor(nqs);
for i=1,nqs,batch_size do
	xlua.progress(i, nqs)
	r=math.min(i+batch_size-1,nqs);
	fv[{{i,r},{}}],qids[{{i,r}}]=forward(i,r);
end



------------------------------------------------------------------------
-- Write to hdf5 file
------------------------------------------------------------------------
print('')
print('Writing output file ...')
paths.mkdir(opt.out_path)
h5=hdf5.open(opt.out_path .. 'questions_features_' .. opt.dataset .. '.h5','w')
h5:write('/ques_indexes',qids)
h5:write('/features',fv)
h5:close()
print('... Done')
