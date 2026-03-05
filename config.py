import torch
import os

class Config:
    """全局配置类"""
    
    # ==================== 路径配置 ====================
    ##San_francisco_C San_francisco_L Hainan_X
    DATA_DIR = '/home/usuaris/csl/shurong.zhang/data/work3_RADAR/data/Hainan_X'
    OUTPUT_DIR = '/home/usuaris/csl/shurong.zhang/data/work3_RADAR/General_decomposition_framework/result/Hainan_X'
    CHECKPOINT_DIR = '/home/usuaris/csl/shurong.zhang/data/work3_RADAR/General_decomposition_framework/checkpoints/Hainan_X'
    
    # ✅ 新增：参考数据目录
    REFERENCE_DIR = '/home/usuaris/csl/shurong.zhang/data/work3_RADAR/data/Hainan_X/reference'
    
    HDR_FILE = os.path.join(DATA_DIR, 'T11.bin.hdr')  # 如果有HDR文件
    DATA_FILES = {
        'T11': 'T11.bin',
        'T22': 'T22.bin',
        'T33': 'T33.bin',
        'T12_real': 'T12_real.bin',
        'T12_imag': 'T12_imag.bin',
        'T13_real': 'T13_real.bin',
        'T13_imag': 'T13_imag.bin',
        'T23_real': 'T23_real.bin',
        'T23_imag': 'T23_imag.bin',
    }
    
    # ==================== 设备配置 ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== 模型配置 ====================
    DECOMP_TYPES = ['3comp', '4comp', '6comp']
    NUM_EXPERTS = {
        '3comp': 1,
        '4comp': 3,
        '6comp': 4,
    }
    
    # ==================== 训练配置 ====================
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_PATCH_SIZE = 128
    DEFAULT_EPOCHS = 100
    DEFAULT_LR = 1e-4
    DEFAULT_WEIGHT_DECAY = 1e-5
    
    # ==================== Loss权重配置 ====================
    DEFAULT_RECON_LAMBDA = 1
    DEFAULT_REFERENCE_LAMBDA = 0.1
    DEFAULT_SMOOTH_LAMBDA = 0.1
    
    
    DEFAULT_SAMPLE_RATIO = 1 
    
    # ==================== 学习率调度器 ====================
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 5
    LR_MIN = 1e-6
    
    # ==================== 早停配置 ====================
    EARLY_MIN_DELTA = 1e-4
    EARLY_PATIENCE = 10
    
    # ==================== 数据加载配置 ====================
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # ==================== 推理配置 ====================
    INFERENCE_BLOCK_SIZE = 128
    INFERENCE_STRIDE = 128
    
    # ==================== 其他可能需要的配置 ====================
   
    OUTLIER_PERCENTILE = 95  # 异常值百分位
    MIN_VALUE = 1e-10  # 最小值限制
    
    @classmethod
    def create_output_dirs(cls):
        """创建必要的输出目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        print(f"✅ 输出目录已创建/验证")
    
    @classmethod
    def validate_paths(cls):
        """验证路径是否存在"""
        if not os.path.exists(cls.DATA_DIR):
            raise FileNotFoundError(f"数据目录不存在: {cls.DATA_DIR}")
        
        # ✅ 新增：验证参考数据目录
        if not os.path.exists(cls.REFERENCE_DIR):
            print(f"⚠️ 参考数据目录不存在: {cls.REFERENCE_DIR}")
            print(f"   将以无监督模式运行（不使用参考值）")
        else:
            print(f"✅ 参考数据目录存在: {cls.REFERENCE_DIR}")
    
    @classmethod
    def get_checkpoint_path(cls, model_type: str) -> str:
        """获取checkpoint路径"""
        return os.path.join(cls.CHECKPOINT_DIR, f'checkpoint_{model_type}.pth')
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("\n" + "="*70)
        print("当前配置")
        print("="*70)
        print(f"数据目录: {cls.DATA_DIR}")
        print(f"输出目录: {cls.OUTPUT_DIR}")
        print(f"Checkpoint目录: {cls.CHECKPOINT_DIR}")
        print(f"参考数据目录: {cls.REFERENCE_DIR}")  # ✅ 新增
        print(f"设备: {cls.DEVICE}")
        print(f"批大小: {cls.DEFAULT_BATCH_SIZE}")
        print(f"Patch尺寸: {cls.DEFAULT_PATCH_SIZE}")
        print(f"训练轮数: {cls.DEFAULT_EPOCHS}")
        print(f"学习率: {cls.DEFAULT_LR}")
        print(f"Loss权重: recon={cls.DEFAULT_RECON_LAMBDA}, "
              f"reference={cls.DEFAULT_REFERENCE_LAMBDA}, "  # ✅ 修改
              f"smooth={cls.DEFAULT_SMOOTH_LAMBDA}")
        print(f"参考值采样: {cls.DEFAULT_SAMPLE_RATIO*100:.0f}%")  # ✅ 新增
        print("="*70)


# ==================== 向后兼容 ====================
# 如果其他代码直接导入这些常量，保持兼容性
DATA_DIR = Config.DATA_DIR
OUTPUT_DIR = Config.OUTPUT_DIR
HDR_FILE = Config.HDR_FILE
DATA_FILES = Config.DATA_FILES