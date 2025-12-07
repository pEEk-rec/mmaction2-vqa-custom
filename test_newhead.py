import torch
import torch.nn as nn

from mmaction.registry import MODELS
from mmaction.utils import SampleList


##########################################################
# 1) Fake backbone output
##########################################################
N = 4
x = torch.randn(N, 768, 8, 7, 7).cuda()


##########################################################
# 2) Fake data_samples
##########################################################
class FakeDataSample:
    def __init__(self, mos, qclass, hallu, light, spatial):
        self.metainfo = dict(
            mos=mos,
            quality_class=qclass,
            hallucination_flag=hallu,
            lighting_flag=light,
            spatial_flag=spatial,
        )

    def set_metainfo(self, d):
        self.metainfo.update(d)


data_samples: SampleList = [
    FakeDataSample(
        mos=50 + i,
        qclass=i % 5,
        hallu=i % 2,
        light=(i + 1) % 2,
        spatial=int(i % 3 == 0),
    ) for i in range(N)
]


##########################################################
# 3) Build the custom head
##########################################################
cfg = dict(
    type="swin_AIHead",   # registered name
    in_channels=768,
    num_classes=5,
    use_uncertainty_weighting=True
)

head: nn.Module = MODELS.build(cfg).cuda()
head.train()

print("\nHead built:", head.__class__.__name__)


##########################################################
# 4) Forward pass
##########################################################
outs = head.forward(x)

print("\n===== Forward Output Shapes =====")
for k, v in outs.items():
    print(k, tuple(v.shape))


##########################################################
# 5) Loss computation
##########################################################
loss_dict = head.loss(x, data_samples)

print("\n===== Loss Dict =====")
for k, v in loss_dict.items():
    if torch.is_tensor(v):
        print(k, float(v.detach().cpu()))
    else:
        print(k, v)


##########################################################
# 6) Backward pass (gradient flow test)
##########################################################
loss = loss_dict["loss"]
loss.backward()
print("\nBackward pass OK")


##########################################################
# 7) UW parameter check
##########################################################
if hasattr(head, "log_vars"):
    print("\nlog_vars:", head.log_vars.detach().cpu())


print("\n✔ ALL TESTS PASSED")
