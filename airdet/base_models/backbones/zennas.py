# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import os, sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_strings = {
    "S16FC1280": "SuperConvK3BNRELU(3,32,2,1)SuperResK3K3(32,128,2,56,1)SuperResK1K3K1(128,240,2,128,1)SuperResK1K5K1(240,360,2,96,1)SuperResK1K3K1(360,336,1,192,1)SuperResK1K5K1(336,1432,2,184,3)SuperResK1K5K1(1432,952,1,184,4)SuperResK1K3K1(952,1712,1,384,3)SuperConvK1BNRELU(1712,1280,1,1)",
}


def _get_model_(arch, num_classes, pretrained=False, opt=None, argv=None, model_name=None,
                plainnet_struct=None, out_indices=(1, 2, 3, 4), use_stage1=False, use_cx2px=False):
    # Any PlainNet
    if arch.find('.py:PlainNet') >= 0:
        module_path = arch.split(':')[0]
        assert arch.split(':')[1] == 'PlainNet'
        my_working_dir = os.path.dirname(os.path.dirname(__file__))
        module_full_path = os.path.join(my_working_dir, module_path)

        import importlib.util
        spec = importlib.util.spec_from_file_location('AnyPlainNet', module_full_path)
        AnyPlainNet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(AnyPlainNet)
        print('Create model: {}'.format(arch))
        # print('string: {}'.format(model_strings[model_name]))
        if plainnet_struct is None:
            model = AnyPlainNet.PlainNet(plainnet_struct=model_strings[model_name], num_classes=num_classes, argv=argv, \
                                         use_detect=True, out_indices=out_indices, use_stage1=use_stage1,
                                         use_cx2px=use_cx2px)
        else:
            model = AnyPlainNet.PlainNet(plainnet_struct=plainnet_struct, num_classes=num_classes, argv=argv, \
                                         use_detect=True, out_indices=out_indices, use_stage1=use_stage1,
                                         use_cx2px=use_cx2px)

    else:
        raise ValueError('Unknown model arch: ' + arch)

    return model


def zennas_model(model_name="S16FC1280", plainnet_struct=None, out_indices=(2, 3, 4, 5), use_stage1=False,
                 num_classes=80, use_cx2px=False):
    masternet_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "PlainNet/masternet.py")
    arch = "%s:PlainNet" % (masternet_path)
    num_classes = num_classes
    # argv = ["", "", "", ""]
    return _get_model_(arch=arch, num_classes=num_classes, argv=None, model_name=model_name, \
                       plainnet_struct=plainnet_struct, out_indices=out_indices, use_stage1=use_stage1,
                       use_cx2px=use_cx2px)


def load_model(model, load_parameters_from, strict_load=False, map_location='cpu'):
    assert os.path.isfile(load_parameters_from), "bad checkpoint to load %s" % (load_parameters_from)
    print('loading params from ' + load_parameters_from)
    checkpoint = torch.load(load_parameters_from, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict_load)

    return model


# from mmcv.runner import BaseModule

# @BACKBONES.register_module()
class ZenNas(nn.Module):
    def __init__(self, net_str=None, out_indices=(2, 3, 4, 5), use_stage1=False, \
                 frozen_stages=-1, zennas_fixbn=False, num_classes=80, init_cfg=None, use_cx2px=False):
        super(ZenNas, self).__init__(init_cfg)

        self.body = zennas_model(plainnet_struct=net_str, \
                                 out_indices=out_indices, use_stage1=use_stage1, num_classes=num_classes,
                                 use_cx2px=use_cx2px)
        self.out_feat_chs = self.body.features_channels_final

        if init_cfg:
            self.body.train()
            self.body = load_model(self.body, init_cfg.checkpoint)
        if zennas_fixbn:
            self.body.freeze_backbone_part(frozen_stages)

        if False:
            from global_utils import export_onnx
            onnx_name = "%s/zennas.onnx" % (cfg.OUTPUT_DIR)
            export_onnx(onnx_name, body, 480, 480, 3, 0, Nboutputs=4)
            exit()

    def forward(self, x):
        """Forward function."""
        return self.body(x)


def export_onnx(model_name="S16FC1280", plainnet_struct=None, onnx_name=None):
    import os, thop
    from global_utils import export_onnx
    import torchvision

    resolution = 480
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # model_name="250M+++"
    # model = zennas_model(model_name)
    model = zennas_model(model_name, plainnet_struct=plainnet_struct)
    model.train()
    pretrain_url = "/mnt3/zhenhong.szh/Projects/NAS/01-NAS/02-NASV4/save_model/run_nas_detection_V100FP16/cnn_laten42e-5_sublayers16_res480_flops20e9_fc1280_N100000/train_120epochs/best-params_rank0.pth"
    model = load_model(model, pretrain_url, strict_load=False, map_location='cpu')

    for name in model.state_dict():
        print(name, model.state_dict()[name].size(), model.state_dict()[name].flatten()[0])
    print("\n%s\n" % str(model))
    print("The num sublayers: %d; FLOPs: %.2f M; Model_size: %.3f M\n" % (
    model.get_num_layers(), model.get_FLOPs(resolution) / 1e6, model.get_model_size() / 1e6))

    if onnx_name is None:
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        onnx_name = os.path.join(root_path, "models/ZenNas/%s/model.onnx" % (model_name))
    os.makedirs(os.path.dirname(onnx_name), exist_ok=True)

    input_h, input_w, input_c = resolution, resolution, 3
    input_D = torch.randn(1, input_c, input_h, input_w)
    export_onnx(onnx_name, model, input_h, input_w, input_c, batch_size=32, \
                export_params=True, export_resolution=True, Nboutputs=4)

    # flops_D, params_D = thop.profile(model, inputs=(input_D, ))
    # flops_D, params_D = thop.clever_format([flops_D, params_D], "%.3f")
    # print('===> decoder:{}flops_{}params\n\n'.format(flops_D, params_D))


if __name__ == "__main__":
    model = zennas_model(model_name="S16FC1280", plainnet_struct=None, out_indices=(3, 4, 5))
    # export_onnx(model_name="S16FC1280", plainnet_struct=None)
