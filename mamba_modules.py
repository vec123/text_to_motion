import torch
import torch.nn as nn
import SSM_modules


class MambaBlock(nn.Module):

    def __init__(self, dim: int, in_channels: int, dt_rank: int, dim_inner: int, d_state: int):

        super(MambaBlock, self).__init__()
        self.proj_one = nn.Linear(dim, dim)
        self.proj_two = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.conv1 = nn.Conv1d(
            in_channels = in_channels, out_channels = in_channels, kernel_size = 3, padding = 1, dilation=1, groups = 1
            )
        self.swish = nn.SiLU()
        self.SSM = SSM_modules.SSM(dim, dt_rank, dim_inner, d_state)

    def forward(self, x):

        skip = x
        print(x.shape)
        x_one = self.proj_one(x)
        print(x_one.shape)
        x_two = self.proj_two(x)

        x_one = self.conv1(x_one)
        print(x_one.shape)
        x_one = self.swish(x_one)
        print(x_one.shape)
        x_one = self.SSM(x_one)
        print(x_one.shape)
        x_two = self.swish(x_two)

        out  = x_one * x_two
        out = out + skip
        out = self.proj_out(out)
        return out


x = torch.rand(1,64,256)

block = MambaBlock(dim = 256, in_channels = 64, dt_rank = 8, dim_inner = 256, d_state = 256)

out = block(x)


#https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py


class HTM(nn.Module):
     
     def __init__(self, input_dim, hidden_dim, output_dim):
        
        self.x_proj = nn.Linear(
           input_dim, hidden_d, bias=False, **factory_kwargs
        )
        self.act = nn.SiLU(),
        self.SSM = SSM_modules.SSM(),

        self.x_aggregate = nn.Linear(
            hidden_d, output_dim, bias=False, **factory_kwargs
        )

     def forward(self, x, num_scans):
        scan_outpus =[]
        for i in range(0,num_scans):

            x_prime = self.proj(x)

            x_prime = self.conv1d(x_prime)

            out = self.SSM(x_prime)

            scan_outpus.append(out)
            
        aggregated_output = torch.stack(scan_outputs, dim=0).mean(dim=0)  
        z_HTM = self.aggregate_linear(aggregated_output)
        return z_HTM
     

class BSM(nn.Module):
     
     def __init__(self, input_dim, hidden_dim, output_dim):
        
        self.x_proj = nn.Linear(
           input_dim, hidden_d, bias=False, **factory_kwargs
        )
        self.act = nn.SiLU(),
        self.SSM = SSM_modules.SSM(),

        self.x_aggregate = nn.Linear(
            hidden_d, output_dim, bias=False, **factory_kwargs
        )

        self.num_scans = 2
        self.gating_weights = nn.Parameter(torch.ones(self.num_scans, requires_grad=True))
        self.softmax = nn.Softmax(dim=0)

     def forward(self, x, directions = ["forward", "backward"]):
    
        for direction in directions:
            if direction == "forward":

                x_prime = self.proj(x)

                x_prime = self.conv1d(x_prime)

                out = self.SSM(x_prime)

                scan_outpus.append(out)

            elif direction == "backward":
                x = x.transpose
                x_prime = self.proj(x)

                x_prime = self.conv1d(x_prime)

                out = self.SSM(x_prime)

            
        scan_outputs_tensor = torch.stack(scan_outputs, dim=0)
        # Define a gating mechanism (e.g., a linear layer followed by a softmax)
        normalized_weights = self.softmax(gating_weights)
        normalized_weights = normalized_weights.view(self.num_scans, 1, 1)  # [num_scans, 1, 1]
        weighted_outputs = scan_outputs_tensor * normalized_weights  # [num_scans, batch_size, features]

        # Sum over scans
        aggregated_output = weighted_outputs.sum(dim=0)  # [batch_size, features]
        z_BSM= self.aggregate_linear(aggregated_output)
        return z_BSM

