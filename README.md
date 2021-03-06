# 한자 고문헌 번역기

BOAZ:elephant: ADV 프로젝트 **한자 문헌 번역기**입니다.

학습 / 검증 데이터셋으로 국사편찬위원회에서 제공하는 [조선왕조실록](http://sillok.history.go.kr/main/main.do) 전문을 사용합니다.

테스트 데이터로는 [승정원일기](http://sjw.history.go.kr/main.do)를 사용합니다.

- 주 사용 기술
  - 모델은 Transformer 를 사용하고 있습니다. ([Attention is All You Need](https://arxiv.org/abs/1706.03762))
  - Subword Tokenizing 
  - Flask 웹 프레임워크
  - Kubernetes

## 번역결과

학습을 진행하지 않은 조선왕조실록 테스트셋과 승정원일기에 대한 [번역결과를 열람](http://bit.ly/boaz-adv-nlp-result)할 수 있습니다.

## 데모

[데모 웹 페이지](http://bit.ly/boaz-adv-nlp-demo)를 운영하고 있습니다. 

## 개발 과정

간략하게나마 [블로그](https://getchan.github.io/projects/adv_pjt_2/)에 개발 과정을 기록했습니다. 

질문이나 피드백 매우 감사합니다. [:e-mail:](mailto:9511chn@gmail.com)