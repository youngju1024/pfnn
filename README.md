
# PFNN 논문 구현

## 2015005014 정영주

>파일 소개
>>PFNN.py
>>>졸업프로젝트의 결과 파일. Default로 08_04_60fps.bvh의 초기정보를 가져온다. 만일 작동이 안되면 bvh풀더 내부의 아무 bvh파일을 드래그 엔드 드롭하면 된다.

>>bvh_viewer_code.py
>>>bvh파일을 드래그 앤드 드롭하면 bvh파일의 애니메이션이 나온다

>>face_vector.py
>>>어깨사이의 벡터와 엉덩이 사이의 벡터의 평균값과 y방향 벡터를 외적하여 motion의 face vector를 저장하는 프로그램

>>find_error.py
>>>NN에서 어디에 가장 오류가 많은지 확인하는 프로그램

>>fps60.py
>>>CMU motion capture data가 120FPS이므로 60FPS로 낮추어주는 프로그램

>>input.py
>>>다른 프로그램으로 만든 input dataset에 필요한 정보들을 모아 input파일로 모으는 프로그램

>>interpolate.py
>>>phase.py로 대략적으로 계산한 phase를 수동으로 보완한 데이터를 사이사이 linear로 보간하여 각 프레임의 phase 값을 정하는 프로그램

>>location.py
>>>frame마다 각각의 joint의 position을 저장하는 프로그램

>>loss_phase.txt
>>>학습할 때 오차가 어느정도 되는지 기록한 텍스트 문서

>>nn.py
>>>pytorch를 이용하여 학습하는 프로그램, CUDA로 학습하도록 설정되어 있다.

>>nn_based_bvh.py
>>>학습결과로 나온 tensor.pt를 이용하여 bvh파일을 드래그 앤드 드롭하면 그 bvh파일의 input 파일과 phase값을 이용하여 결과값을 애니메이션으로 나타내는 프로그램

>>output.py
>>>다른 프로그램으로 만든 output dataset에 필요한 정보들을 모아 output파일로 모으는 프로그램

>>phase.py
>>>양 발뒷굼치의 속도와 위치를 이용하여 대략적인 phase값의 기준점(0,pi,2pi)를 자동적으로 잡아주는 프로그램

>>tensor.py
>>>nn.py로 학습한 결과 파일

>>trajectory.py
>>>trajectory의 위치를 계산하는 프로그램

>>vel.py
>>>각 프레임의 각각의 joint의 velocity를 구하는 프로그램