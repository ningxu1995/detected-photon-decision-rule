"""
FIG 2 -- the photon-budget machinery, three thematic columns:

  Col 1  Budget law & density   N >~ N_vox * p * kappa / SNR^2
         (a) density crossover  (b) regime phase diagram
  Col 2  Axial DOF / missing cone
         (c) coherent-vs-confocal transfer split map   (d) kappa_axial(NA)
  Col 3  Spot-array encoding knob
         (e) super-oscillation cost eta_enc(gamma)   (f) photons-for-resolution front

All three cores are the validated computations from the standalone scripts,
re-rendered into one figure with unified styling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from numpy.linalg import inv
from scipy.linalg import cho_factor, cho_solve, inv as scipy_inv
from scipy.special import j0

FL='#6a1b9a'; LF='#1565c0'                       # fluorescence / label-free colors
lam=0.5; NA=1.4; n_med=1.515; SNRst=10.0; beta=3.0; deps=0.05
kj=2*np.pi/lam; A_b=0.42*(n_med/NA)*lam**2
cell=lam/(2*NA)


def inverse_regularized(A, eps=1e-12):
    """Fast inverse for the small regularised Fisher matrices used here."""
    Areg=A+eps*np.eye(A.shape[0])
    try:
        c=cho_factor(Areg, lower=True, check_finite=False)
        return cho_solve(c, np.eye(A.shape[0]), check_finite=False)
    except Exception:
        return scipy_inv(Areg, check_finite=False)

# ============================================================
# CORE 1 : multivoxel Fisher -- density crossover + regime map
# ============================================================
def compute_density():
    """Compute the density crossover/regime map.

    Optimised without changing the underlying model: label-free Fisher matrices
    are evaluated once per bead layout at ci=1 and then rescaled analytically by
    ci(d)^-2 for each bead diameter.
    """
    Ng=56; pitch=cell*10/Ng; cen=Ng//2
    fx=np.fft.fftshift(np.fft.fftfreq(Ng,d=pitch)); FX,FY=np.meshgrid(fx,fx,indexing='ij')
    pup=(np.hypot(FX,FY)<=NA/lam).astype(complex)
    hc=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pup))); hc/=np.sqrt((np.abs(hc)**2).sum())
    hF=np.abs(hc)**2; hF/=hF.max(); gc=hc**2; gc/=np.abs(gc).max()
    def mkscan(s):
        sp=max(1,int(round(s/pitch))); idx=np.arange(0,Ng,sp)
        X,Y=np.meshgrid(idx,idx,indexing='ij'); return np.stack([X.ravel(),Y.ravel()],1)
    scanF=mkscan(cell/2); scanL=mkscan(cell/2); NL_=scanL.shape[0]
    Ncell=100; hc_px=0.7*cell/pitch
    def cols(beads,psf,scan):
        Nb=beads.shape[0]; M=np.zeros((scan.shape[0],Nb),dtype=psf.dtype)
        sx=beads[:,0][None,:]-scan[:,0][:,None]+cen
        sy=beads[:,1][None,:]-scan[:,1][:,None]+cen
        ok=(sx>=0)&(sx<Ng)&(sy>=0)&(sy<Ng)
        sx=np.clip(sx,0,Ng-1).astype(int); sy=np.clip(sy,0,Ng-1).astype(int)
        M[ok]=psf[sx[ok],sy[ok]]
        return M
    def place(rho,seed):
        # Fast hard-core placement: shuffle all valid pixels once and greedily
        # accept candidates. This avoids long rejection loops at high densities.
        r=np.random.default_rng(seed); Nb=max(1,int(round(rho*Ncell))); C=[]
        xs=np.arange(3,Ng-3); ys=np.arange(3,Ng-3)
        cand=np.array(np.meshgrid(xs,ys,indexing='ij')).reshape(2,-1).T
        r.shuffle(cand)
        for x,y in cand:
            ok=True
            for (xx,yy) in C:
                if (x-xx)**2+(y-yy)**2<hc_px**2: ok=False; break
            if ok:
                C.append((int(x),int(y)))
                if len(C)>=Nb: break
        return np.array(C)
    def ci_(d): V=(4*np.pi/3)*(d/2)**3; return kj*V/A_b
    def NF(beads):
        H=cols(beads,hF,scanF).real; Nb=beads.shape[0]; s=np.ones(Nb)
        sig=H@s; b=beta*sig.mean(); mu=sig+b; W=1/mu
        FIM=(H*W[:,None]).T@H; C=inverse_regularized(FIM, 1e-12)
        rel=np.sqrt((np.diag(C)/s**2).mean()); N0=(rel*SNRst)**2
        return N0*mu.sum()
    def NL_base_ci1(beads):
        # ci=1 base; final photon count rescales as 1/ci(d)^2.
        Gc=cols(beads,gc,scanL); F=0.5*Gc; R=F.conj().T@F; Nb=beads.shape[0]
        Ruu=4*np.real(R); Ruv=4*np.imag(R); FIM=np.block([[Ruu,Ruv],[Ruv.T,Ruu]])
        Ci=inverse_regularized(FIM, 1e-12)
        rel=np.sqrt(((np.diag(Ci)[:Nb]+np.diag(Ci)[Nb:])/deps**2).mean())
        return (rel*SNRst)**2*NL_
    def NL_from_base(base,d):
        return base/(ci_(d)**2)

    nreal=4; rhos_t=np.geomspace(0.02,1.0,12); dl=[0.6,0.7,0.9]
    rho_ax=np.zeros(len(rhos_t)); NFr=np.zeros(len(rhos_t)); NLr=np.zeros((3,len(rhos_t)))
    for ri,rt in enumerate(rhos_t):
        nf=[]; nl=[[],[],[]]; ra=[]
        for s in range(nreal):
            b=place(rt,100*ri+s); ra.append(len(b)/Ncell); nf.append(NF(b))
            base=NL_base_ci1(b)
            for di,d in enumerate(dl): nl[di].append(NL_from_base(base,d))
        rho_ax[ri]=np.mean(ra); NFr[ri]=np.mean(nf)
        for di in range(3): NLr[di,ri]=np.mean(nl[di])
    def xover(NFc,NLc):
        f=np.log10(NFc/NLc)
        return float(np.interp(0,f,rho_ax)) if f[0]<0<f[-1] else np.nan
    rstar=[xover(NFr,NLr[di]) for di in range(3)]
    rg=np.geomspace(0.03,1.0,18); dg=np.linspace(0.2,1.0,18); ratio=np.zeros((18,18))
    ci2=np.array([ci_(d)**2 for d in dg])
    for j,rt in enumerate(rg):
        nf=[]; nl_scaled=[]
        for s in range(3):
            b=place(rt,7000+100*j+s); nf.append(NF(b))
            base=NL_base_ci1(b)
            nl_scaled.append(base/ci2)
        ratio[:,j]=np.log10(np.mean(nf)/np.mean(nl_scaled,axis=0))
    return rho_ax,NFr,NLr,dl,rstar,rg,dg,ratio

# ============================================================
# CORE 2 : 3D transfer functions -- missing cone + kappa_axial
# ============================================================
def build_grid(N,ext):
    ax=np.linspace(-ext,ext,N); dk=ax[1]-ax[0]
    KX,KY,KZ=np.meshgrid(ax,ax,ax,indexing='ij'); KR=np.sqrt(KX**2+KY**2+KZ**2)
    return ax,dk,KX,KY,KZ,KR
def cap(NA_,KZ,KR,dk):
    al=np.arcsin(NA_/n_med); shell=np.abs(KR-n_med)<0.75*dk; cz=KZ>=n_med*np.cos(al)-1e-9
    C=np.zeros(KZ.shape); sel=shell&cz; C[sel]=np.sqrt(np.clip(KZ[sel]/n_med,1e-6,1)); return C,al
def autocorr(C):
    F=np.fft.fftn(np.fft.ifftshift(C)); return np.fft.fftshift(np.real(np.fft.ifftn(np.abs(F)**2)))
def autoconv(O):
    F=np.fft.fftn(np.fft.ifftshift(O)); return np.fft.fftshift(np.real(np.fft.ifftn(F**2)))
def compute_missingcone():
    N,ext=116,6.6; ax,dk,KX,KY,KZ,KR=build_grid(N,ext); c=N//2
    C,al0=cap(NA,KZ,KR,dk); Owf=autocorr(C); Owf/=Owf.max(); Ocf=autoconv(Owf); Ocf/=Ocf.max()
    g0=np.pi/2-al0; Mwf=Owf[c,:,:].T; Mcf=Ocf[c,:,:].T
    def metrics(NAx,Ns=104,exts=6.6):
        axs,dks,kx,ky,kz3,kr=build_grid(Ns,exts); cc=Ns//2
        Cn,al=cap(NAx,kz3,kr,dks); Ow=autocorr(Cn); Ow/=Ow.max(); Oc=autoconv(Ow); Oc/=Oc.max()
        prof=Oc[cc,cc,:]; pos=axs>1e-9; p0=prof[pos]; kp0=axs[pos]
        kf=np.linspace(kp0.min(),kp0.max(),600); pf=np.clip(np.interp(kf,kp0,p0),0,None); pf/=pf.max()
        band=pf>0.10; kc=kf[band].max() if band.any() else 0.0
        dkf=kf[1]-kf[0]; sel=kf<=kc
        kap=kc/(np.sum(pf[sel]**2)*dkf) if kc>0 else np.nan
        dz=np.pi/(kc*(2*np.pi/lam))*1e3 if kc>0 else np.nan
        return kap,dz
    NAs=np.linspace(0.6,1.45,14); kap=[]; dz=[]
    for x in NAs:
        k,d=metrics(x); kap.append(k); dz.append(d)
    return ax,Mwf,Mcf,g0,NAs,np.array(kap),1-NAs/n_med,np.array(dz)

# ============================================================
# CORE 3 : super-oscillation envelope + encoding Pareto front
# ============================================================
def compute_encoding():
    Nrho=320; rho=np.linspace(0,1,Nrho); drho=rho[1]-rho[0]
    Nr=640; r=np.linspace(1e-4,8.0,Nr); dr=r[1]-r[0]
    M=j0(2*np.pi*np.outer(r,rho))
    def focal(P):
        E=M@(P*rho)*drho; return E,E**2
    Pair=np.ones_like(rho); Ea,Ia=focal(Pair)
    fwhm_airy=2*r[np.where(Ia<0.5*Ia[0])[0][0]]
    b_list=np.linspace(0.30,0.95,24); w_list=np.linspace(-1.5,0.95,50); pts=[]
    for b in b_list:
        inner=rho<=b
        for w in w_list:
            P=np.where(inner,1.0,w); E,I=focal(P); I0=I[0]
            if I0<=0: continue
            h=np.where(I<0.5*I0)[0]
            if not h.size: continue
            fwhm=2*r[h[0]]; sc=np.where(np.diff(np.sign(E))!=0)[0]; rz=r[sc[0]] if sc.size else r[-1]
            Ecen=np.sum(I[r<=rz]*r[r<=rz])*dr; Etot=np.sum(I*r)*dr
            if Etot<=0: continue
            g=fwhm/fwhm_airy; e=Ecen/Etot
            if 0.42<g<1.25 and 1e-4<e<=1.0: pts.append((g,e))
    pts=np.array(pts); gb=np.linspace(0.45,1.05,26); eg=[]; ee=[]
    for i in range(len(gb)-1):
        sel=(pts[:,0]>=gb[i])&(pts[:,0]<gb[i+1])
        if sel.any(): eg.append(0.5*(gb[i]+gb[i+1])); ee.append(pts[sel,1].max())
    eg=np.array(eg); ee=np.array(ee)
    for i in range(len(ee)-2,-1,-1): ee[i]=min(ee[i],ee[i+1])
    airy_fwhm_nm=fwhm_airy*lam*1000/NA; abbe_nm=lam*1000/(2*NA)
    def eta(g):
        g=np.atleast_1d(g).astype(float); o=np.empty_like(g)
        lo=g<=eg[0]; hi=g>=1.0; mid=~lo&~hi
        o[mid]=np.exp(np.interp(g[mid],eg,np.log(ee))); o[hi]=0.857
        o[lo]=ee[0]*(g[lo]/eg[0])**6; return o
    NF0=525.0; NL0=1530.0; eta0=0.857
    gF=np.linspace(0.49,1.20,160); NF=NF0*(1/gF**2)*(eta0/eta(gF)); resF=gF*airy_fwhm_nm
    gL=np.linspace(1.0,1.30,60); NL=NL0*(1/gL**2); resL=gL*airy_fwhm_nm
    mcg=[0.52,0.68]; mcres=[g*airy_fwhm_nm for g in mcg]
    mcN=[NF0*(1/g**2)*(eta0/float(eta(g)[0])) for g in mcg]
    return eg,ee,resF,NF,resL,NL,abbe_nm,airy_fwhm_nm,mcg,mcres,mcN,NF0,NL0

print("computing core 1 (density)...");      D = compute_density()
print("computing core 2 (missing cone)..."); Mc = compute_missingcone()
print("computing core 3 (encoding)...");      Enc = compute_encoding()
print("rendering...")

rho_ax,NFr,NLr,dl,rstar,rg,dg,ratio = D
axk,Mwf,Mcf,g0,NAs,kap,fmiss,dzc = Mc
eg,ee,resF,NF,resL,NL,abbe_nm,airy_fwhm_nm,mcg,mcres,mcN,NF0,NL0 = Enc

# ============================================================
#                       RENDER  (2 x 3)
# ============================================================
# PRL-friendly sans-serif typography.  Arial is requested; Helvetica/DejaVu
# are included as robust fallbacks on systems where Arial is unavailable.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 15,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig=plt.figure(figsize=(18.2,10.4))
gs=GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.34,top=0.875,bottom=0.085,left=0.065,right=0.975)
axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[1,0])
axC=fig.add_subplot(gs[0,1]); axD=fig.add_subplot(gs[1,1])
axE=fig.add_subplot(gs[0,2]); axF=fig.add_subplot(gs[1,2])

# ---- column headers ----
# The former global top line is split into compact column-level subtitles so it
# no longer sits above or collides with the column headings.
fig.text(0.18,0.956,'Budget law & density',ha='center',fontsize=18,weight='bold')
fig.text(0.18,0.925,r'$N \gtrsim N_{\rm vox}\,p\,\kappa/{\rm SNR}^{2}$',ha='center',fontsize=15)
fig.text(0.515,0.956,'Axial DOF — missing cone',ha='center',fontsize=18,weight='bold')
fig.text(0.515,0.925,r'NA=1.4, $\lambda$=500 nm, $n$=1.515',ha='center',fontsize=15)
fig.text(0.85,0.956,'Spot-array encoding knob',ha='center',fontsize=18,weight='bold')
fig.text(0.85,0.925,r'$\delta\epsilon$=0.05, SNR*=10',ha='center',fontsize=15)

def tag(ax,t,color='k'):
    # Panel labels are intentionally omitted; they will be added separately.
    return

def apply_tick_style(ax):
    ax.tick_params(axis='both',which='major',labelsize=14.5,width=0.9,length=4)
    ax.tick_params(axis='both',which='minor',width=0.7,length=2.5)

# ---------- (a) density crossover ----------
cols=['#0a8f6b','#1565c0','#c84b00']
axA.plot(rho_ax,NFr,color=FL,lw=2.8,label=f'fluorescence (β={int(beta)})',zorder=5)
for di,(d,c) in enumerate(zip(dl,cols)):
    axA.plot(rho_ax,NLr[di],color=c,lw=2.3,label=f'label-free {int(d*1000)} nm')
    if not np.isnan(rstar[di]):
        yc=np.interp(rstar[di],rho_ax,NFr); axA.plot(rstar[di],yc,'o',color=c,ms=7.5,mec='k',mew=0.75,zorder=6)
axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xlabel('fill factor ρ'); axA.set_ylabel('photons to SNR*=10')
axA.set_title('density crossover',pad=8); axA.grid(True,which='both',alpha=0.2)
axA.legend(loc='lower right',frameon=True,fontsize=12,borderpad=0.4,labelspacing=0.3,handlelength=2.2)
tag(axA,'a')
axA.text(0.018, NFr[0]*1.5, r'fluor $\propto$ $\rho$', color=FL, fontsize=14, rotation=22)
# Moved upward between the blue and red label-free curves.
axA.text(0.018,2.1e4,'label-free relatively flat',color=LF,fontsize=14,ha='left',va='center')
apply_tick_style(axA)

# ---------- (b) regime phase diagram ----------
im=axB.contourf(rg,dg*1000,ratio,levels=np.linspace(-2,2,21),cmap='RdBu_r',extend='both')
axB.contour(rg,dg*1000,ratio,levels=[0],colors='k',linewidths=2.3)
axB.set_xscale('log'); axB.set_xlabel('fill factor ρ'); axB.set_ylabel('bead diameter (nm)')
axB.set_title(r'regime map   $\log_{10}(N_F/N_L)$',pad=8)
cb=plt.colorbar(im,ax=axB,fraction=0.046,pad=0.03,ticks=[-2,0,2]); cb.ax.tick_params(labelsize=16)
cb.set_label('fluorescence cheaper ←  → label-free cheaper',fontsize=12)
# Region labels capitalised and repositioned; both are white as requested.
stroke=[pe.withStroke(linewidth=1.4, foreground='black', alpha=0.35)]
axB.text(0.62,915,'Label-free',color='w',fontsize=14,ha='center',va='center',path_effects=stroke)
axB.text(0.035,270,'Fluorescence',color='w',fontsize=14,ha='left',va='center',path_effects=stroke)
tag(axB,'b')
apply_tick_style(axB)

# ---------- (c) missing-cone split map ----------
teal=LinearSegmentedColormap.from_list('t',['#04121a','#0a3d52','#1f7fa6','#7fd4e6','#f2fbff'])
KY=axk; KZ=axk
combined=np.where(KY[None,:]<0, Mwf, Mcf)   # left half = coherent O_wf, right = confocal O_cf
ext=[axk[0],axk[-1],axk[0],axk[-1]]
imc=axC.imshow(10*np.log10(np.clip(combined,1e-3,None)),origin='lower',extent=ext,cmap=teal,vmin=-30,vmax=0,aspect='auto')
zz=np.linspace(-1.5,1.5,40); tG=np.tan(g0)
axC.plot(zz*tG, zz,'w--',lw=1.4); axC.plot(-zz*tG, zz,'w--',lw=1.4)   # cone boundary (X)
axC.axvline(0,color='w',lw=0.8,alpha=0.5)
axC.set_xlim(-3,3); axC.set_ylim(-1.5,1.5)
axC.set_xlabel(r'$K_\perp$ ($k_0$)'); axC.set_ylabel(r'$K_z$ ($k_0$)')
axC.set_title('coherent (Left) vs confocal-fluor (Right)',pad=8)
# Moved to approximately Kz=-1.25 and made bold.
axC.text(-1.75,-1.24,'coherent\nmissing cone',color='w',fontsize=13.5,ha='center',va='center')
axC.text(1.62,-1.24,'confocal\nfilled',color='w',fontsize=13.5,ha='center',va='center')
cb2=plt.colorbar(imc,ax=axC,fraction=0.046,pad=0.03,ticks=[-30,-15,0]); cb2.set_label('dB',fontsize=14); cb2.ax.tick_params(labelsize=14)
tag(axC,'c',color='w')
apply_tick_style(axC)

# ---------- (d) kappa_axial(NA) ----------
axD.plot(NAs,kap,'o-',color='#0a3d52',lw=2.4,ms=5,label=r'Fluorescence $\kappa_{\rm axial}$ (cone filled)')
axD.set_yscale('log'); axD.set_ylim(1,52)
axD.set_xlabel('NA'); axD.set_ylabel(r'Fluorescence $\kappa_{\rm axial}$',color='#0a3d52')
axD.tick_params(axis='y',labelcolor='#0a3d52'); axD.set_title('axial conditioning',pad=8)
axD.grid(True,which='both',alpha=0.2)
axb=axD.twinx(); axb.plot(NAs,100*fmiss,'s--',color='#c0392b',lw=2.0,ms=4.5,label='missing-cone fraction')
axb.set_ylabel('label-free missing cone (%)',color='#c0392b',fontsize=16.5); axb.tick_params(axis='y',labelcolor='#c0392b',labelsize=15); axb.set_ylim(0,70)
l1,la1=axD.get_legend_handles_labels(); l2,la2=axb.get_legend_handles_labels()
legD=axD.legend(l1+l2,la1+la2,fontsize=11,loc='upper right',frameon=True,
                title=r'$n$=1.515, $\lambda$=500 nm',title_fontsize=10,
                borderpad=0.45,labelspacing=0.3,handlelength=2.0)
# Representative confocal-fluorescence axial-resolution estimates.
# Small arrowed callouts keep the three NA-specific δz values anchored to the
# fluorescence axial-conditioning curve without visually crowding the panel.
# Shorter, lighter δz callouts: first two below the curve, the high-NA
# estimate above it.  The arrows are kept close to the representative points.
for x,scale,va in [(0.7,0.74,'top'),
                   (1.0,0.74,'top'),
                   (1.4,1.30,'bottom')]:
    i=np.argmin(np.abs(NAs-x))
    axD.annotate(r'$\delta z$=' + f'{dzc[i]:.0f} nm',
                 xy=(NAs[i],kap[i]), xycoords='data',
                 xytext=(NAs[i], kap[i]*scale), textcoords='data',
                 fontsize=10.5,color='#0a3d52',ha='center',va=va,
                 bbox=dict(boxstyle='round,pad=0.16',facecolor='white',
                           edgecolor='#0a3d52',alpha=0.88,linewidth=0.55),
                 arrowprops=dict(arrowstyle='-|>',color='#0a3d52',lw=0.68,
                                 shrinkA=2,shrinkB=2,mutation_scale=5.4))
tag(axD,'d')
apply_tick_style(axD)

# ---------- (e) encoding cost eta_enc ----------
axE.axvspan(0.42,0.70,color='#fdebd0',alpha=0.7)
axE.plot(eg,ee,'o-',color='#117a65',lw=2.4,ms=4.2,label=r'$\eta_{\rm enc}$ (Toraldo)')
axE.plot(1.0,0.848,'k*',ms=13); axE.text(1.015,0.50,'0.84 Airy',fontsize=12)
axE.axvspan(0.52,0.68,color=LF,alpha=0.12)
axE.text(0.60,0.72,'MCoSM \ndemonstrated \nspot regime [22]',fontsize=10.5,color='#08519c',ha='center',va='center')
axE.text(0.472,20.8e-3,'super-oscillation wall',fontsize=12,color='#b9770e',
         ha='center',va='center',rotation=90)
axE.set_yscale('log'); axE.set_ylim(2e-3,1.7); axE.set_xlim(0.45,1.18)
axE.set_xlabel(r'spot size $\gamma$ = FWHM/Airy'); axE.set_ylabel(r'central-lobe $\eta_{\rm enc}$')
axE.set_title('super-oscillation cost',pad=8); axE.grid(True,which='both',alpha=0.2)
axE.legend(fontsize=12,loc='lower right',frameon=True,borderpad=0.4,labelspacing=0.3,handlelength=2.2)
tag(axE,'e')
apply_tick_style(axE)

# ---------- (f) photons-for-resolution Pareto front ----------
axF.axvspan(70,abbe_nm,color='#eaecee',alpha=0.8)
axF.axvline(abbe_nm,color='k',ls='--',lw=1.3)
axF.text(abbe_nm*1.02,2.0e4,'Abbe',fontsize=12,rotation=90,va='bottom',ha='left')
axF.text(abbe_nm*1.05,1.5e4,r'($\lambda$/2NA)',fontsize=11,rotation=90,va='bottom',ha='left')
axF.plot(resF,NF,color=FL,lw=2.8,label='fluorescence (p=1)')
axF.plot(resL,NL,color=LF,lw=2.8,label='label-free (p=2)')
axF.plot(abbe_nm,NL0,'s',color=LF,ms=8.5,mec='k',mew=0.75)
axF.text(abbe_nm*1.035,NL0*1.24,'wall',fontsize=11,color=LF,ha='left',va='bottom')
axF.plot(mcres,mcN,'o',color=FL,ms=7.5,mec='k',mew=0.75)
# Repositioned labels to avoid the border and the purple curve.
label_positions=[(mcres[0]-17, mcN[0]*0.62, '0.52 Airy', 'left'),
                 (mcres[1]-1, mcN[1]*0.67, '0.68 Airy', 'right')]
for x,y,txt,ha in label_positions:
    axF.text(x,y,txt,fontsize=10,color=FL,ha=ha,va='top')
# Purple callout for the super-oscillation photon wall.
idx=np.argmin(np.abs(resF-120))
axF.annotate('super-oscillation\nphoton wall', xy=(resF[idx],NF[idx]), xycoords='data',
             xytext=(145,1.25e5), textcoords='data', fontsize=10.5, color=FL,
             ha='center',va='center',
             arrowprops=dict(arrowstyle='->', color=FL, lw=1.5, shrinkA=2, shrinkB=2))
axF.set_yscale('log'); axF.set_xlim(80,250); axF.set_ylim(1.5e2,3e5)
axF.set_xlabel('lateral resolution (nm)'); axF.set_ylabel('photons / cell')
axF.set_title('photons-for-resolution',pad=8); axF.grid(True,which='both',alpha=0.2)
axF.legend(fontsize=10.5,loc='upper right',frameon=True,borderpad=0.4,labelspacing=0.3,handlelength=2.2)
axF.text(95,5.0e2,'coherent passband wall',fontsize=11.5,color='#566573',ha='left',va='bottom')
axF.text(95,2.3e2,'(label-free: nonlinear \nfar-field super-oscillation)',fontsize=11.5,color='#566573',ha='left',va='bottom')
axF.text(185,2.3e2,'(coarser → cheaper)',fontsize=11.5,color='#566573',ha='left',va='bottom')
tag(axF,'f')
apply_tick_style(axF)

# Save a high-resolution PNG.  Low PNG compression keeps saving fast and
# avoids timeout on slower environments.
out_png='Fig2_microadjusted_v2.png'
plt.savefig(out_png,dpi=300,pil_kwargs={'compress_level':1})
print(f"saved {out_png}", flush=True)
plt.close(fig)
print(f"crossover rho* (600/700/900nm): {[f'{x:.2f}' for x in rstar]}")
print(f"kappa_axial range: {kap.min():.1f}-{kap.max():.1f}; missing-cone frac {fmiss[0]*100:.0f}%->{fmiss[-1]*100:.0f}%")
