## MobileBERT를 활용한 멕도날드 매장 리뷰 감성분석 프로젝트
---
## 1. 개요
---

패스트푸드 프랜차이즈는 소비자와의 접점이 많은 업종으로, 고객 리뷰는 매장 서비스 품질과 브랜드 이미지에 직접적인 영향을 미친다. 
그중에서도 멕도날드는 전 세계적으로 가장 많은 매장을 운영하는 브랜드 중 하나이며, 국내에서도 주요 상권마다 매장을 보유하고 있다.
이러한 리뷰 데이터를 분석함으로써 고객 불만 요인을 파악하고, 서비스 품질 개선의 방향성을 도출할 수 있다.

본 프로젝트에서는 멕도날드 매장에 대한 온라인 리뷰를 분석하여 감성(긍정/부정)을 분류하고, 부정 리뷰가 집중된 항목을 정리해보려 한다.
이를 통해 브랜드 이미지 관리 및 운영 개선에 도움이 되는 인사이트를 도출하고자 한다.

---
## 2. 데이터

[[데이터 원본]](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews)

### 2-1 데이터 구성

| reviewer_id | store_name | category | store_address | latitude | longitude | rating_count | review_time | review | rating |
|-------------|-------|--------|------------|---------|---------|------------|-------------|--------|--------|
| 고유 식별자      | 매장명   | 카테고리   | 매장 주소      | 위도      | 경도      | 평점개수       | 리뷰날짜        | 리뷰     | 평점     |

데이터 수는 33,096건이 있고, 평점은 1점부터5점까지 구성되어있다.

### 2-2 데이터 부가정보

<img src="img1.png" width="600" />

총 39개의 매장에서 조사한 리뷰들이다.

<img src="img.png" width="600" />

리뷰 문장 길이 500자 이하 그래프이다.
10자 리뷰가 가장 많이 나타난다.

---
## 3. 데이터 전처리

MobileBERT를 활용한 리뷰 기반 감성 분류 모델을 학습시키기 위해 모델 입력에 불필요한 정보를 제거하고 핵심 텍스트(review)와 레이블(rating)만 남겨 데이터 전처리를 수행하였다.
이를 통해 학습 효율성과 성능 향상을 도모하였다.

'rating'에서 1,2점은 0(부정) 4,5점은 1(긍정)으로 라벨링 하였고, 3점은 제거하였다.

| |review|label|
|-|------|--------|
|1|Why does it look like someone spit on my food?...|0|
|2|It'd McDonalds. It is what it is as far as the food and atmosphere go...|1|
|..|...|..|
|28511|It's good, but lately it has become very expensive...|1|
|28512|they took good care of me|1|

전테 데이터 33,096건에서 렌덤으로 긍정 1000건, 부정1000건만 가져와서 모델을 학습시켰다.

| |review|label|
|-|------|--------|
|1|Shitty service shitty employees|0|
|2|not bad ehh....|1|
|..|...|..|
|1999|rude|0|
|2000|The People Who Work There Were Very Rude&Short In Response...|0|

---
## 4. 결과

### 4-1 MobileBERT 학습 결과
<img src="img2.png" width="600" />

---
## 5. 결론 및 느낀점

