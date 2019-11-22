# 한자 문헌 번역기

BOAZ ADV 프로젝트 **한자 문헌 번역기**입니다.

학습 데이터셋으로 국사편찬위원회에서 제공하는 [조선왕조실록](http://sillok.history.go.kr/main/main.do) 전문을 사용합니다.


사용 모델은 LSTM기반의 Sequence to Sequence를 사용하고 있습니다.

![모델 구조 사진](./prototype/plot_model.png)

> 추후 BERT 등의 모델로 시도해볼 예정


테스트 데이터로는 [승정원일기](http://sjw.history.go.kr/main.do)를 사용합니다.


## 데모

데모 API로 실행해 볼 수 있습니다.

`34.85.53.48:5000/demo` 에 `POST`방식으로 `input`인자를 넣어 request하면 됩니다.

[`httpie`](https://www.popit.kr/introduce_httpie/)를 이용하여 작동해본 예시입니다.

> python의 경우 `pip install httpie`로 간단히 설치할 수 있습니다.

```shell
$ http -v POST 34.85.53.48:5000/demo input='司憲府劾王康嘗體察三道, 擾民作弊。'
```

```shell
>> 형조/NNP에/JKB전지/NNG하/XSV기/ETN를/JKO,/SP"/SS지금/MAG듣/VV건대/EC,/SP_의/JKG_는/JX모두/MAG_하/XSV아/EC,/SP_을/JKO_하/XSV아/EC_하/XSV아/EC_하/XSV게/EC하/VV았/EP으니/EC,/SP이것/NP은/JX비록/MAG작/VA은/ETM일/NNB이/VCP라도/EC또한/MAJ다/MAG알/VV지/EC
```