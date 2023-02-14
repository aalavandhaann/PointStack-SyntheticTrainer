device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")  

net.to(device)  
if (device.type == 'cuda') and (ngpu > 1):     
    net = nn.DataParallel(net, list(range(ngpu)))