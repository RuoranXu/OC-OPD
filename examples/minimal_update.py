"""A synthetic OC-OPD optimizer update; no model downloads or evaluation."""

import torch
from oc_opd import surrogate

torch.manual_seed(0)
student = torch.nn.Linear(16, 64)
teacher = torch.nn.Linear(16, 64).requires_grad_(False)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
features = torch.randn(2, 8, 16)
response_mask = torch.ones(2, 8, dtype=torch.bool)
response_mask[:, :3] = False
with torch.no_grad():
    teacher_logits = teacher(features)
loss, metrics = surrogate(student(features), teacher_logits, response_mask)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
print({name: values.mean().item() for name, values in metrics.items()})
