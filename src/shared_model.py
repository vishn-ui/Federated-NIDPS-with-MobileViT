import torch
import timm

def get_mobilevit_64bit():
    # Load the hardware-optimized XXS version
    model = timm.create_model('mobilevit_xxs', pretrained=True, num_classes=2)
    
    # CRITICAL: Convert all parameters and buffers to 64-bit (Double Precision)
    model = model.to(torch.float64)
    
    return model