from model import Net
from perforatedai import utils_perforatedai as UPA

def count_params(model):
    return sum(p.numel() for p in model.parameters())

baseline = Net()
dendritic = UPA.initialize_pai(Net())

b = count_params(baseline)
d = count_params(dendritic)

print("Baseline parameters:", b)
print("Dendritic parameters:", d)
print("Reduction %:", round(100 * (1 - d / b), 2))
