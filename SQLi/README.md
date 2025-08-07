# SQLi공격에 대한 WGAN-CP
현재 총 2차 수정을 가함.
사용한 DB는 [SQL injection Dataset](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset)에서 다운받았으며, 위의 데이터를 일반화 처리를 걸쳐 사용함
| 여부 | 종류 |
| :---: | --- |
| 일반화 O | 텍스트, 공백문자, 따옴표, SQL 명령어 |
| 일반화 X | 라이브러리명, 컬럼명을 포함한 데이터베이스 설정에 따라 달라지는 요소 |
