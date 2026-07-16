import numpy as np, onnxruntime as ort, torch
from df.enhance import init_df, enhance, df_features
from df.io import get_test_sample
from df.utils import as_complex
SR,NB_DF,DF_ORDER,LA,NB_ERB = 48000,96,5,2,32
O="/tmp/dfn3_onnx/tmp/export"
enc=ort.InferenceSession(f"{O}/enc.onnx"); erbd=ort.InferenceSession(f"{O}/erb_dec.onnx"); dfd=ort.InferenceSession(f"{O}/df_dec.onnx")
model, dfst, _ = init_df()
erb_fb = np.asarray(dfst.erb_widths())      # widths per erb band (sums to F)
a = get_test_sample(SR).reshape(1,-1).float()
ref = enhance(model, dfst, a).numpy().reshape(-1)
# proven feature extraction (torch)
spec, feat_erb, feat_spec = df_features(a, dfst, NB_DF)   # spec[1,1,Tf,F,2], erb[1,1,Tf,32], spec_feat[1,2,Tf,96]
spec_np = spec.numpy(); Tf = spec_np.shape[2]
e0,e1,e2,e3,emb,c0,lsnr = enc.run(None,{"feat_erb":feat_erb.numpy(),"feat_spec":feat_spec.numpy()})
m = erbd.run(None,{"emb":emb,"e3":e3,"e2":e2,"e1":e1,"e0":e0})[0]        # [1,1,Tf,32]
coefs = dfd.run(None,{"emb":emb,"c0":c0})[0]
# spec complex [1,1,Tf,F]
sc = spec_np[...,0] + 1j*spec_np[...,1]                                   # [1,1,Tf,F]
# ERB mask -> freq gains by repeating each band across its width
gains = np.repeat(m[0,0], erb_fb.astype(int), axis=-1)                    # [Tf, F]
sc_m = sc * gains[None,None]
# deep filter on low bins
c = coefs.reshape(1,Tf,DF_ORDER,NB_DF,2); cc=(c[...,0]+1j*c[...,1]).transpose(0,2,1,3)  # [1,O,Tf,96]
sp = sc_m[0,0,:,:NB_DF]                                                   # [Tf,96]
spp = np.pad(sp,((DF_ORDER-1-LA,LA),(0,0)))
frames = np.stack([spp[i:i+Tf] for i in range(DF_ORDER)],axis=-1)        # [Tf,96,O]
out = np.einsum("tfn,ntf->tf", frames, cc[0])                            # [Tf,96]
sc_e = sc_m.copy(); sc_e[0,0,:,:NB_DF]=out
# synthesis via proven df_state (needs [1,Tf,F,2] real)
se = np.stack([sc_e[0].real, sc_e[0].imag],axis=-1).astype(np.float32)   # [1,Tf,F,2]
audio_out = np.asarray(dfst.synthesis(se[...,0]+1j*se[...,1])).reshape(-1)
n=min(len(ref),len(audio_out)); r,o=ref[:n],audio_out[:n]
print(f"MSE {np.mean((r-o)**2):.2e}  corr {np.corrcoef(r,o)[0,1]:.5f}")
print("PARITY_PASS" if np.corrcoef(r,o)[0,1]>0.99 else "PARITY_CLOSE" if np.corrcoef(r,o)[0,1]>0.9 else "PARITY_FAIL")
