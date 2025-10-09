import torch
import torch.nn as nn

# class Model(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.W=nn.Parameter(torch.zeros(1,1))
#     self.b=nn.Parameter(torch.zeros(1))

#   def forward(self,x):
#     y=x@self.W+self.b
#     return y
  
class Model(nn.Module):
  def __init__(self,input_size=1,output_size=1):
    super().__init__()
    self.linear=nn.Linear(input_size,output_size)
    #print("self.linear.W:",self.linear.W)

  def forward(self,x):
    y=self.linear(x)
    return y

model=Model()

for param in model.parameters():
  print(param)

input_tensor=torch.rand(4,1)
output_tensor=model.linear(input_tensor)

print("model.linear:",model.linear)
print("output_tensor=",output_tensor)
#
# Linear layer 는 PyTorch에서 신경망의 완전 연결(fully connected) 레이어를 구현하는 데 사용되는 클래스
# 입력 데이터에 대해 선형 변환(linear transformation)을 수행하여, 이를 다음 레이어로 전달하는 역할
#
# # Linear 레이어 정의
# linear_layer = nn.Linear(in_features=5, out_features=2)
# # linear_layer = nn.Linear(in_features=5, out_features=2, bias=True)

# # 입력 텐서 생성
# input_tensor = torch.randn(1, 5)

# # 선형 변환 적용
# output_tensor = linear_layer(input_tensor)
# print(output_tensor)