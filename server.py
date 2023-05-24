import json
import argparse
import datetime
import logging
import logging.config
import copy
import collections
import os
import random

import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
# GPT2LMHeadModel, BertModel, MT5ForConditionalGeneration
import tornado.ioloop
import tornado.httpclient
import tornado.escape
import tornado.web

from config import global_config as cfg
from main import Model
from reader import read_data, convert_eval_batch_turn, batch_align

RESP_BODY = {
	"userid": "",
	"outparams": {
		'resp':'',
		'type':'',
		'goal':'',
		'spoken':[''],
		'kb':{}
	}
}

# need to be improved, global variable will encounter bug while interacting with multiple uses

class MainHandler(tornado.web.RequestHandler):
	# need to be improved, global variable will encounter bug while interacting with multiple uses
	# a dict coordinating with the user id will do
	# spoken = []
	# kb = []

	def initialize(self, dial_sys, retrieve_sys, dataset):
		self.system = dial_sys
		self.system.model.eval()
		self.retrieval_model = retrieve_sys
		self.retrieval_model.eval()
		self.bert_tokenizer = BertTokenizer.from_pretrained(cfg.bert_save_path)
		# dataset is the dataset used for demo, i.e., the test of mobileCS
		self.dataset = dataset
		# dialog history, do not use yet
		self.pv_batch=None
		self.cls = dial_sys.tokenizer.convert_tokens_to_ids('[CLS]')

	def prepare(self):
		#logging.info('req_headers: %s' % dict(self.request.headers))
		self.req_body = json.loads(self.request.body) if self.request.body else {}
		self.set_header(name='Content-Type', value='application/json; charset=UTF-8')
	
	def get(self):
		self.write('Dialog System')
	
	def post(self):
		resp_body=copy.deepcopy(RESP_BODY)
		eos_a_id=self.system.tokenizer.convert_tokens_to_ids('[EOS_SI]')
		eos_r_id=self.system.tokenizer.convert_tokens_to_ids('[EOS_S]')
		sep_id=self.system.tokenizer.convert_tokens_to_ids('[SEP]')
		max_len_a=20
		max_len_resp=90
		# start a new dialog, clear up the memory, 
		# randomly select a dialog from the test sets and return the goal to the use
		if self.req_body['type']=='new':
			# fix the seed if neccessary
			dial = random.sample(self.dataset, k=1)
			resp_body.update({"userid": self.req_body['userid']})
			resp_body['outparams'].update(
				{'goal':dial[0][0]['org_goal'],
				'type':self.req_body['type'],
				'kb': dial[0][0]['KB']
				}
			)
		elif self.req_body['type']=='1':
			# 用户输入
			spoken = self.req_body['spoken']
			kb = self.req_body['kb']
			with torch.no_grad():
				user = self.req_body['content']
				turn_batch = {}
				turn_batch['用户'] = [user]
				turn_batch['user'] = [self.system.tokenizer.encode('[EOS_L]' + user +'[EOS_U]')[1:-1]]
				turn_batch['resp'] = [[]]
				# get history
				spoken, hist = self.system.get_spoken(spoken, turn_batch['用户'], role='user')

				# get db results
				db_gen,db_decode = self.system.query_KB_retrieval(self.retrieval_model, self.bert_tokenizer, spoken, hist, [kb])
				# o_batch = self.system.generate_batch_turn_level(i_batch, ground_truth_db=cfg.gt_db)
				# can not reuse the generate function due to the session level setting

				contexts = convert_eval_batch_turn(self.cls, turn_batch, self.pv_batch, mode='gen_ar', 
							db_gen=db_gen)
				#remove bspn_gen=bs_gen,ent_gen=entity_gen to simplify the dependency

				inputs,attentions = batch_align(contexts,return_attn=True)
				inputs = torch.tensor(inputs).to(self.system.model.device)
				attentions = torch.tensor(attentions).to(self.system.model.device)
				#aspn_batch=self.system.generate_batch(self.model, contexts, max_len_a, eos_a_id)
				resp_batch=self.system.generate_batch(self.system.model, contexts, max_len_resp, eos_r_id)

				#aspn_gen = self.get_xspn(aspn_batch, eos_a_id,sep_id)
				resp_gen = self.system.get_xspn(resp_batch, eos_r_id,sep_id,eos_a_id)

                # turn_batch['kb'] = self.decode_batch(turn_batch['aspn'])
				# turn_batch['客服'] = self.decode_batch(turn_batch['resp'])
				# turn_batch['客服意图-生成']= self.decode_batch(aspn_gen)
				turn_batch['resp']=self.system.decode_batch(resp_gen)

				spoken, _=self.system.get_spoken(spoken, turn_batch['resp'], role='system')

				resp_body.update({"userid": self.req_body['userid']})
				resp_body['outparams'].update(
					{'resp':turn_batch['resp'],
					'type':self.req_body['type'],
					'spoken':spoken,
					'kb':kb}
				)
				logging.info(self.req_body)
				logging.info(resp_body)
		elif self.req_body['type']=='end':
			resp_body['outparams'].update(
					{'type':self.req_body['type']}
				)
		self.write(json.dumps(resp_body))
	

if __name__ == "__main__":
	# settings, modify the model path here (gpt_path)
	port=59998
	cfg.gpt_path = 'experiments/jsa_4_scratch/best_model'
	cfg.device = [0]
	cfg.gt_db = False
	cfg.retrieve_kb = True

	cfg._init_logging_handler() # cfg._init_logging_handler(port)
	# Dialog system
	logging.info('Loading dialog system...')

	DS = Model(cfg.device)

	retriever = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)
	retriever.to(cfg.device[0])
	dataset = read_data(DS.tokenizer, return_dict=True)['test']
	logging.info('Dialog systems loaded.')
	# app
	application = tornado.web.Application([(r"/", MainHandler, dict(dial_sys=DS, retrieve_sys=retriever, dataset=dataset)),])
	application.listen(port)
	logging.info('listening on 127.0.0.1:%s...' % port)
	tornado.ioloop.IOLoop.current().start()
