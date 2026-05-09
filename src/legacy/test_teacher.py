"""Smoke tests para os novos Teachers. Executar com: uv run python src/test_teacher.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
from utils.config import *

def test_ptv3_compatible_block_shapes():
    from utils.architectures import PTv3CompatibleBlock
    block = PTv3CompatibleBlock(d_model=128, n_heads=8, k_neighbors=16)
    N = 256
    x   = torch.randn(N, 128)
    xyz = torch.randn(N, 3)
    out = block(x, xyz)
    assert out.shape == (N, 128), f"Esperado (256,128), obtido {out.shape}"
    print("✓ PTv3CompatibleBlock: shape OK")

def test_ptv3_compatible_teacher_shapes():
    from utils.architectures import PTv3CompatibleTeacher
    teacher = PTv3CompatibleTeacher(input_dim=15, d_model=128, checkpoint_path=None)
    teacher.eval()
    N = 128
    x = torch.randn(N, 15)
    with torch.no_grad():
        bottleneck = teacher(x)
    assert bottleneck.shape == (N, 512), f"bottleneck: esperado (128,512), obtido {bottleneck.shape}"
    feats = {}
    h1 = teacher.feature_adapter.register_forward_hook(lambda m,i,o: feats.update({'adapter': o}))
    h2 = teacher.lfa.register_forward_hook(lambda m,i,o: feats.update({'lfa': o}))
    h3 = teacher.blocks[0].register_forward_hook(lambda m,i,o: feats.update({'blk0': o}))
    with torch.no_grad():
        teacher(x)
    h1.remove(); h2.remove(); h3.remove()
    assert feats['adapter'].shape == (N, 128), f"adapter: {feats['adapter'].shape}"
    assert feats['lfa'].shape     == (N, 128), f"lfa: {feats['lfa'].shape}"
    assert feats['blk0'].shape    == (N, 128), f"blk0: {feats['blk0'].shape}"
    print("✓ PTv3CompatibleTeacher: shapes e hooks OK")

def test_ptv3_compatible_teacher_freeze():
    from utils.architectures import PTv3CompatibleTeacher
    teacher = PTv3CompatibleTeacher(input_dim=15, checkpoint_path=None)
    teacher.freeze()
    trainable = [n for n, p in teacher.named_parameters() if p.requires_grad]
    frozen    = [n for n, p in teacher.named_parameters() if not p.requires_grad]
    assert len(trainable) > 0, "Nenhum parâmetro treinável após freeze"
    assert all('feature_adapter' in n for n in trainable), \
        f"Params treináveis inesperados: {trainable}"
    print(f"✓ PTv3CompatibleTeacher freeze: {len(frozen)} frozen, {len(trainable)} treináveis")

def test_ptv3_teacher_import():
    try:
        import torchsparse
        from utils.architectures import PTv3Teacher
        teacher = PTv3Teacher(input_dim=15, checkpoint_path=None)
        N = 64
        x = torch.randn(N, 15)
        with torch.no_grad():
            bottleneck = teacher(x)
        assert bottleneck.shape == (N, 512), f"bottleneck: {bottleneck.shape}"
        print("✓ PTv3Teacher (torchsparse): shapes OK")
    except ImportError:
        print("⚠  PTv3Teacher: torchsparse não instalado — pulando (comportamento esperado)")

def test_build_teacher_fallback():
    """Verifica que build_teacher() retorna Teacher válido com forward correto."""
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "ts", os.path.join(os.path.dirname(__file__), "teacher_student_v1.py"))
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)

    teacher = ts.build_teacher(input_dim=15, ptv3_ckpt=None, s3dis_ckpt=None)
    N = 64
    x = torch.randn(N, 15)
    with torch.no_grad():
        bottleneck = teacher(x)
    assert bottleneck.shape == (N, 512), f"bottleneck: {bottleneck.shape}"
    print(f"✓ build_teacher: {type(teacher).__name__} retornou (N,512)")

def test_teacher_student_integration():
    """Verifica forward pass completo com o Teacher ativo via build_model()."""
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "ts", os.path.join(os.path.dirname(__file__), "teacher_student_v1.py"))
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ts.build_model(device)
    model.eval()

    N = 512
    x = torch.randn(N, 15, device=device)

    with torch.no_grad():
        out = model(x)

    assert 'teacher_scales' in out
    assert 'student_scales' in out
    assert 'bottleneck'     in out

    t3, t2, t1 = out['teacher_scales']
    s3, s2, s1 = out['student_scales']

    assert t3.shape == (N, 256), f"t3: {t3.shape}"
    assert t2.shape == (N, 128), f"t2: {t2.shape}"
    assert t1.shape == (N, 64),  f"t1: {t1.shape}"
    assert s3.shape == (N, 256), f"s3: {s3.shape}"
    assert s2.shape == (N, 128), f"s2: {s2.shape}"
    assert s1.shape == (N, 64),  f"s1: {s1.shape}"

    teacher_name = type(model.teacher).__name__
    print(f"✓ Integração: {teacher_name} | t3{tuple(t3.shape)} t2{tuple(t2.shape)} t1{tuple(t1.shape)}")

if __name__ == '__main__':
    test_ptv3_compatible_block_shapes()
    test_ptv3_compatible_teacher_shapes()
    test_ptv3_compatible_teacher_freeze()
    test_ptv3_teacher_import()
    test_build_teacher_fallback()
    test_teacher_student_integration()
    print("TODOS OS TESTES PASSARAM")
