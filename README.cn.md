# COmfyUI LatentSync 

## 介绍

ComfyUI_LatentSync 是一个对 [LatentSync](https://github.com/bytedance/LatentSync) 的comfy ui自定义节点封装.  

## 功能
    1. 调整了符合comfy ui规则的统一模型存放位置.  
    2. 独立了模型加载节点, 符合comfy ui节点运行缓存规则, 在运行中无需重复加载(之前有大佬做的[ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper),每次运行就会清掉模型,再次运行会重新加载).  
    3. 添加了insight face连线, 现在你可以使用现有的insight face节点,而无需再重复下载一套冗余的insight face模型了  

## 如何使用
    1. cd ComfyUI/custom_nodes/  
    2. git clone https://github.com/HJH-AILab/ComfyUI_LatentSync.git  
    3. cd ComfyUI_LatentSync  
    4. 安装[LatentSync](https://github.com/bytedance/LatentSync)依赖  
    5. 配置 ComfyUI/extra_model_paths.yaml  
        ```yaml
        comfyui:
            LatentSync: &lt;your_latentsync_models_path>
        ```
    6. 下载模型到&lt;your_latentsync_models_path>目录下  