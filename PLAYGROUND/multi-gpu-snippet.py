# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")  

# net.to(device)  
# if (device.type == 'cuda') and (ngpu > 1):     
#     net = nn.DataParallel(net, list(range(ngpu)))

color_choices = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #3
                     (1, 0, 0), (0, 1, 0), #5
                     (1, 0, 0), (0, 1, 0), #7
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #11
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #15
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), # 18
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #21
                     (1, 0, 0), (0, 1, 0), #23
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #27
                     (1, 0, 0), (0, 1, 0), #29
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), # 35
                     (1, 0, 0), (0, 1, 0), #37
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), #40
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), #43
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), #46
                     (1, 0, 0), (0, 1, 0), (0, 0, 1), #49
]

print(len(color_choices))