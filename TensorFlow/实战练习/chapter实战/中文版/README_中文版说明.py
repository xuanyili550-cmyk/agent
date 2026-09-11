"""
================================================================================
 中文版 · 说明与中文模型选型（面试/生产都用得上）
================================================================================
 为什么要中文版：模型只认它训练过的语种/词表，英文模型喂中文会失准。做中文系统，
 每个环节都要换成中文(或多语言)模型。本文件夹是关键案例的中文模型版，全部实测跑通。

 —— 文件 ——
   中文_情感分析.py        uer/roberta-base-finetuned-dianping-chinese
   中文_语义检索RAG.py      BAAI/bge-small-zh-v1.5（注意 CLS 池化 + 查询前缀）
   中文_NER实体识别.py      uer/roberta-base-finetuned-cluener2020-chinese（逐字，要去空格）
   中文_智能客服全流程.py    三个中文模型协作(情感+NER+检索)

 —— 中文模型选型速查(按环节)——
   环节        常用中文模型
   情感/分类   uer/roberta-*-dianping/jd-chinese、IDEA-CCNL/Erlangshen-*、多语言 lxyuan/distilbert-*-sentiments
   句向量/检索 BAAI/bge-large/base/small-zh-v1.5、shibing624/text2vec-base-chinese、m3e-base
   NER 实体    uer/*-cluener2020-chinese、shibing624/bert4ner-base-chinese、UIE(通用信息抽取)
   通用/生成   Qwen(通义千问)、GLM(智谱)、百川、DeepSeek、Yi —— 中文 LLM 首选
   多语言(中英都要) paraphrase-multilingual-MiniLM、bge-m3、distilbert-multilingual-sentiments

 —— 中文特有的坑(面试常问)——
   1) bge 系列用 CLS 池化(取 [:,0])，不是 mean 池化；检索时“查询”加前缀提示提升召回。
   2) 有些中文 NER 是“逐字”的，输出词里字间有空格，要 .replace(' ','') 拼回。
   3) 中文实体类型体系不同(CLUENER 有 人名/地址/公司/机构/职位/景点/书名... 10类)。
   4) 中文分词/大小写/繁简：垂直领域(医疗/金融/法律)最好用领域模型或自己微调(UIE/LoRA)。
   5) 评估集也要中文的，别拿英文基准评中文模型。

 —— 一句话 ——
   做中文 NLP：模型语种必须和数据语种一致；不确定就用“多语言模型”。生成走中文 LLM(Qwen/GLM)。
================================================================================
"""
print(__doc__)
