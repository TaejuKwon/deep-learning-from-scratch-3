# **역전파 알고리즘**

수치미분 -> 계산량이 너무 많음 -> 역전파 알고리즘
gradient checking: 역전파와 수치미분의 값을 비교하는 것 (역전파는 버그가 있기 쉬움)

1. 연쇄 법칙 (Chain Rule)
   
합성함수의 미분: 구성 함수 각각을 미분한 후 곱한 것과 같다

$\frac{dy}{dx} = \frac{dy}{dy}\frac{dy}{db}\frac{db}{da}\frac{da}{dx}$
