from requests import post
from copy import deepcopy
import random, json
import argparse
REQ_BODY = {
    'userid': '',
    'type':'1',
    'content':'',
    'spoken':'',
	'kb':{}
}
WINDOWN_SIZE=900
PAD_SIZE=15

def input_kb():
    try:
        kb = ''
        while True:
            temp = input()
            temp = temp.strip()
            if temp == 'done' or temp == 'quit':
                break
            kb += temp
        kb_dict = eval(kb)
    except:
        print('Wrong format, please input again.')
        print("Please remember that input 'done' after input the KB to continue.")
        if temp == 'quit':
            return {}
        kb_dict = input_kb()
    return kb_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # the ip should be set to the ip of the server
    parser.add_argument('--url', default='http://101.6.68.85:60006')
    # server10 'http://101.6.68.173:60005'
    args = parser.parse_args()
    server_url=args.url
    userid=random.randint(1,1000)
    while(1):
        text=input('输入 NEW 开始新的测试 (服务地址：{}, 用户ID:{})'.format(server_url, userid))
        if text.lower()=='new':
            req_body=deepcopy(REQ_BODY)
            req_body['userid']=userid
            req_body['type']='new'
            resp = post(url=server_url, json=req_body)
            #print(resp.content)
            resp_body = resp.json() if resp.content else {}
            break
    while(resp_body['outparams']['type']!='end'):
        """
        if resp_body['type']=='KB':
            print(resp_body['resp'])
            KB=input_kb()
            req_body['type']='KB'
            req_body['content']=KB
        elif resp_body['type']=='start':
            print('***对话开始（输入END结束）***')
            user=input('用户:')
            if user.lower()=='end':
                req_body['type']='end'
            else:
                req_body['type']='utterance'
                req_body['content']=user
        elif resp_body['type']=='resp':
            print('系统:', resp_body['resp'])
            user=input('用户:')
            if user.lower()=='end':
                req_body['type']='end'
            else:
                req_body['type']='utterance'
                req_body['content']=user
        """
        if resp_body['outparams']['type']=='new':
            print('***对话开始（输入END结束）***')
            print(f"你的目标：{resp_body['outparams']['goal']}")
            user=input('用户:')
            req_body['type'] = '1'
            req_body['content'] = user
            req_body['kb'] = resp_body['outparams']['kb']
            req_body['spoken'] = resp_body['outparams']['spoken']
        elif resp_body['outparams']['type']=='1':
            print('系统:', resp_body['outparams']['resp'])
            user=input('用户:')
            if user.lower()=='end':
                req_body['type']='end'
            else:
                req_body['type']='1'
                req_body['content']=user
                req_body['kb'] = resp_body['outparams']['kb']
                req_body['spoken'] = resp_body['outparams']['spoken']
        else:
            print('Unknown response type!')
        resp = post(url=server_url, json=req_body)
        resp_body = resp.json() if resp.content else {}
        

    