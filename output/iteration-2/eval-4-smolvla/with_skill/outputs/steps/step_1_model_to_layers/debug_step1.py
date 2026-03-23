import sys, torch, math
sys.path.insert(0, 'C:/Users/liang/decomposition_workspace/decompose-workspace/iteration-2/eval-4-smolvla/with_skill/outputs/level_3_model')
import smolvla as om
sys.path.insert(0, 'C:/Users/liang/decomposition_workspace/decompose-workspace/iteration-2/eval-4-smolvla/with_skill/outputs/steps/step_1_model_to_layers')
import refactored as rm

torch.manual_seed(42)
o = om.Model(); o.eval()
r = rm.RefactoredModel(); r.eval()

# Transfer weights
osd = o.state_dict(); rsd = r.state_dict()
for k in rsd:
    if k in osd and osd[k].shape == rsd[k].shape:
        rsd[k] = osd[k].clone()
r.load_state_dict(rsd)

torch.manual_seed(42)
ins = om.get_inputs()
pv, lt, lm, st, ac, no, tm = ins

with torch.no_grad():
    # Compare prefix embedding
    o_pre, o_pm, o_am = o.embed_prefix(pv, lt, lm, st)

    # Manually replicate refactored prefix
    r_img = r.connector(r.vision_encoder(pv))
    r_img = r_img * torch.tensor(r_img.shape[-1]**0.5, dtype=r_img.dtype, device=r_img.device)
    r_lang = r.text_embed(lt) * math.sqrt(960)
    r_state = r.state_proj(st)[:, None, :]
    r_pre = torch.cat([r_img, r_lang, r_state], dim=1)

    # Detailed image debugging
    o_img_raw = o.vision_encoder(pv)
    r_img_raw = r.vision_encoder(pv)
    print(f"Vision enc diff: {(o_img_raw - r_img_raw).abs().max().item():.2e}")

    o_img_conn = o.connector(o_img_raw)
    r_img_conn = r.connector(r_img_raw)
    print(f"Connector diff: {(o_img_conn - r_img_conn).abs().max().item():.2e}")

    o_img_scaled = o_img_conn * torch.tensor(o_img_conn.shape[-1]**0.5, dtype=o_img_conn.dtype)
    r_img_scaled = r_img * 1  # already scaled above
    print(f"Img scaled diff: {(o_img_scaled - r_img_scaled).abs().max().item():.2e}")

    # Compare individual components
    print(f"Lang diff: {(o.embed_language_tokens(lt)*math.sqrt(960) - r_lang).abs().max().item():.2e}")
    print(f"State diff: {(o.state_proj(st)[:,None,:] - r_state).abs().max().item():.2e}")

    print(f"Prefix diff: {(o_pre - r_pre).abs().max().item():.2e}")
    print(f"Prefix shape: o={o_pre.shape}, r={r_pre.shape}")

    # Compare suffix embedding
    time_exp = tm[:, None, None]
    x_t = time_exp * no + (1 - time_exp) * ac

    o_suf, o_spm, o_sam = o.embed_suffix(x_t, tm)

    r_ae = r.action_in_proj(x_t)
    r_te = om.create_sinusoidal_pos_embedding(tm, 720, 4e-3, 4.0, device='cpu')
    r_te = r_te.type(dtype=r_ae.dtype)[:, None, :].expand_as(r_ae)
    r_at = torch.cat([r_ae, r_te], dim=2)
    r_suf = r.action_time_mlp(r_at)

    print(f"Suffix diff: {(o_suf - r_suf).abs().max().item():.2e}")
    print(f"Suffix shape: o={o_suf.shape}, r={r_suf.shape}")

    # Compare mask assembly
    o_padm = torch.cat([o_pm, o_spm], dim=1)
    o_attm = torch.cat([o_am, o_sam], dim=1)
    print(f"o_am dtype: {o_am.dtype}, o_sam dtype: {o_sam.dtype}")
    print(f"o_attm dtype: {o_attm.dtype}")

    # Now test the full forward
    o_out = o(*ins)
    r_out = r(*ins)
    print(f"Full output diff: {(o_out - r_out).abs().max().item():.2e}")
