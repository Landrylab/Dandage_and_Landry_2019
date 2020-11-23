# Figure EV1
## panel A
### plot#1
def plot_hist_dcs_cs_mean_cs1_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not(plotp="plot/hist_dcs_cs_mean_cs1_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.svg",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/hist_dcs_cs_mean_cs1_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.tsv");
    params_saved=yaml.load(open(f"plot/hist_dcs_cs_mean_cs1_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    #     dplot=dplot.sort_values(by=params['colsubsets'])
    from rohan.dandage.plot.dist import hist_annot
    if ax is None:ax=plt.subplot()
    ax=hist_annot(dplot,ax=ax,**params)
    return ax
## panel B
### plot#1
def plot_hist_dcs_cs_mean_cs2_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not(plotp="plot/hist_dcs_cs_mean_cs2_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.svg",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/hist_dcs_cs_mean_cs2_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.tsv");
    params_saved=yaml.load(open(f"plot/hist_dcs_cs_mean_cs2_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    #     dplot=dplot.sort_values(by=params['colsubsets'])
    from rohan.dandage.plot.dist import hist_annot
    if ax is None:ax=plt.subplot()
    ax=hist_annot(dplot,ax=ax,**params)
    return ax
## panel C
### plot#1
def plot_hist_dcs_cs_mean_cs2_1_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not(plotp="plot/hist_dcs_cs_mean_cs2_1_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.svg",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/hist_dcs_cs_mean_cs2_1_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.tsv");
    params_saved=yaml.load(open(f"plot/hist_dcs_cs_mean_cs2_1_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    #     dplot=dplot.sort_values(by=params['colsubsets'])
    from rohan.dandage.plot.dist import hist_annot
    if ax is None:ax=plt.subplot()
    ax=hist_annot(dplot,ax=ax,**params)
    return ax
## panel D
### plot#1
def plot_hist_dcs_cs_mean_cs3_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not(plotp="plot/hist_dcs_cs_mean_cs3_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.svg",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/hist_dcs_cs_mean_cs3_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.tsv");
    params_saved=yaml.load(open(f"plot/hist_dcs_cs_mean_cs3_essential_or_notnon_essential_or_notdriver_or_notoncogene_or_nottumor_suppressor_or_not.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    #     dplot=dplot.sort_values(by=params['colsubsets'])
    from rohan.dandage.plot.dist import hist_annot
    if ax is None:ax=plt.subplot()
    ax=hist_annot(dplot,ax=ax,**params)
    return ax
# Figure 1
## panel A
### plot#1
def plot_dist_paralog_or_singleton(plotp="plot/dist_paralog_or_singleton",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_paralog_or_singleton.tsv");
    params_saved=yaml.load(open(f"plot/dist_paralog_or_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    params_ax={'ylim':[-0.5,1],
    }
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params,params_ax=params_ax)
    return ax
## panel B
### plot#1
def plot_scatter_cs_paralog_or_singleton_paralog_singleton(plotp="plot/scatter_cs_paralog_or_singleton_paralog_singleton",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_cs_paralog_or_singleton_paralog_singleton.tsv");
    params_saved=yaml.load(open(f"plot/scatter_cs_paralog_or_singleton_paralog_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.scatter import plot_scatterbysubsets
    if ax is None:ax=plt.subplot()
    ax=plot_scatterbysubsets(df=dplot,
    **params,
    colannot='dataset',
    dfout=False,label_n=True,
    test=False,ax=ax)    
    return ax
## panel C
### plot#1
def plot_scatter_cs_paralog_or_singleton_paralog_singleton_paralog_unclassified(plotp="plot/scatter_cs_paralog_or_singleton_paralog_singleton_paralog_unclassified",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_cs_paralog_or_singleton_paralog_singleton_paralog_unclassified.tsv");
    params_saved=yaml.load(open(f"plot/scatter_cs_paralog_or_singleton_paralog_singleton_paralog_unclassified.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.scatter import plot_scatterbysubsets
    if ax is None:ax=plt.subplot()
    ax=plot_scatterbysubsets(df=dplot,
    **params,
    colannot='dataset',
    dfout=False,label_n=True,
    test=False,ax=ax)    
    return ax
## panel D
### plot#1
def plot_dist_evolutionary_distance_essential_genes_in_paralog(plotp="plot/dist_evolutionary_distance_essential_genes_in_paralog",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_evolutionary_distance_essential_genes_in_paralog.tsv");
    params_saved=yaml.load(open(f"plot/dist_evolutionary_distance_essential_genes_in_paralog.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    if ax is None:ax=plt.subplot()
    params_ax={'xlabel':''}
    ax=plot_dist_comparison(dplot,
    params_ax=params_ax,
    ax=ax,**params)
    levels=dplot['taxonomy level'].value_counts().head(10).index.tolist()
    df=dplot.loc[:,['taxonomy level','evolutionary distance']].drop_duplicates().dropna(how='any')
    df=df.loc[(df['taxonomy level'].isin(levels)),:]
    # df['bin']=pd.cut(df['evolutionary distance'],bins=12)
    df['bin']=pd.qcut(df['evolutionary distance'],q=8)
    df['bin mid']=df['bin'].apply(lambda x :x.left)
    df=df.groupby('bin mid').agg({'taxonomy level':lambda x : ', '.join(x)})
    d=df.to_dict()['taxonomy level']
    _=[ax.text(ax.get_xlim()[1]*1.05,y,d[y]) for y in d]
    return ax
# Figure 2
## panel A
### plot#1
def plot_dist_heteromer_or_not__all_ppi__mean(plotp="plot/dist_heteromer_or_not__all_ppi__mean",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_heteromer_or_not__all_ppi__mean.tsv");
    params_saved=yaml.load(open(f"plot/dist_heteromer_or_not__all_ppi__mean.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    params_ax={'ylim':[-0.5,1],
    }        
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params,params_ax=params_ax)
    #     ax.set_title(colparalog_subset)
    return ax
## panel B
### plot#1
def plot_scatter_cs_heteromer_or_not__all_biogrid__heteromer_not_heteromer(plotp="plot/scatter_cs_heteromer_or_not__all_biogrid__heteromer_not_heteromer",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_cs_heteromer_or_not__all_biogrid__heteromer_not_heteromer.tsv");
    params_saved=yaml.load(open(f"plot/scatter_cs_heteromer_or_not__all_biogrid__heteromer_not_heteromer.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.scatter import plot_scatterbysubsets
    if ax is None:ax=plt.subplot()
    ax=plot_scatterbysubsets(df=dplot,
    **params,
    colannot='dataset',
    dfout=False,label_n=True,
    test=False,ax=ax)    
    return ax
## panel C
### plot#1
def plot_scatter_cs_heteromer_or_homomer__all_biogrid__heteromer_and_homomer_homomer(plotp="plot/scatter_cs_heteromer_or_homomer__all_biogrid__heteromer_and_homomer_homomer",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_cs_heteromer_or_homomer__all_biogrid__heteromer_and_homomer_homomer.tsv");
    params_saved=yaml.load(open(f"plot/scatter_cs_heteromer_or_homomer__all_biogrid__heteromer_and_homomer_homomer.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.scatter import plot_scatterbysubsets
    if ax is None:ax=plt.subplot()
    ax=plot_scatterbysubsets(df=dplot,
    **params,
    colannot='dataset',
    dfout=False,label_n=True,
    test=False,ax=ax)    
    return ax
## panel D
### plot#1
def plot_dist_ppi_type_ds_heteromer_or_not(plotp="plot/dist_ppi_type_ds_heteromer_or_not",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_ppi_type_ds_heteromer_or_not.tsv");
    params_saved=yaml.load(open(f"plot/dist_ppi_type_ds_heteromer_or_not.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    params_ax={
    #     'ylim':[-0.5,1],
    }        
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params,params_ax=params_ax)
    return ax
## panel E
### plot#1
def plot_plot_heteromer_or_not_cell_line_wise_summary(plotp="plot/plot_heteromer_or_not_cell_line_wise_summary",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/plot_heteromer_or_not_cell_line_wise_summary.tsv");
    params_saved=yaml.load(open(f"plot/plot_heteromer_or_not_cell_line_wise_summary.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    ax=dplot.plot(ax=ax,cmap='Set1')
    ax.legend(bbox_to_anchor=[1.05,1.1],title='age group (significance)')
    ax.set_ylabel('CS median')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['heteromer','not heteromer'])
    ax.set_xlim(-0.25,1.25)
    return ax
# Figure 4
## panel A
### plot#1
def plot_heatmap_correlations_xyz_summary(plotp="plot/heatmap_correlations_xyz_summary",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/heatmap_correlations_xyz_summary.tsv");
    params_saved=yaml.load(open(f"plot/heatmap_correlations_xyz_summary.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.heatmap import annot_heatmap
    from rohan.dandage.plot.annot import pval2annot
    if ax is None:ax=plt.subplot()
    dheatmap=dplot.pivot_table(columns=['colx'],index=['y\n{covariate}'],values='r')
    dannot=dplot.pivot_table(columns=['colx'],index=['y\n{covariate}'],values='p-val').applymap(lambda x : pval2annot(x,fmt='<'))
    ax=sns.heatmap(dheatmap,
    cmap='coolwarm',vmin=-0.5,vmax=0.5,cbar_kws={'label':'$\\rho$'},ax=ax)
    ax=annot_heatmap(ax, dannot.T, xoff=0, yoff=0, kws_text={'va':'center'}, 
    annot_left='(', annot_right=')', annothalf='upper',
    )
    ax.set_xlabel('x')
    return ax
## panel B
### plot#1
def plot_dist___of_interactions_heteromer_or_not__all_ppi_(plotp="plot/dist___of_interactions_heteromer_or_not__all_ppi_",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist___of_interactions_heteromer_or_not__all_ppi_.tsv");
    params_saved=yaml.load(open(f"plot/dist___of_interactions_heteromer_or_not__all_ppi_.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params)
    return ax
## panel C
### plot#1
def plot_dist_fpkm__log2_scale__heteromer_or_not__all_ppi_(plotp="plot/dist_fpkm__log2_scale__heteromer_or_not__all_ppi_",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_fpkm__log2_scale__heteromer_or_not__all_ppi_.tsv");
    params_saved=yaml.load(open(f"plot/dist_fpkm__log2_scale__heteromer_or_not__all_ppi_.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params)
    return ax
## panel D
### plot#1
def plot_swarm_feature_importances_pcorr_heteromer_or_not_heteromer_or_not__all_ppi_(plotp="plot/swarm_feature_importances_pcorr_heteromer_or_not_heteromer_or_not__all_ppi_",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/swarm_feature_importances_pcorr_heteromer_or_not_heteromer_or_not__all_ppi_.tsv");
    params_saved=yaml.load(open(f"plot/swarm_feature_importances_pcorr_heteromer_or_not_heteromer_or_not__all_ppi_.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    ax=sns.swarmplot(data=dplot,dodge=False,  
    size=10,ax=ax,**params['swarmplot'])
    ax.legend(frameon=True)
    return ax
## panel E
### plot#1
def plot_plot_classifier_deleteriousness_metrics_essential_or_not__cs_mean__cs_mean(plotp="plot/plot_classifier_deleteriousness_metrics_essential_or_not__cs_mean__cs_mean",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/plot_classifier_deleteriousness_metrics_essential_or_not__cs_mean__cs_mean.tsv");
    params_saved=yaml.load(open(f"plot/plot_classifier_deleteriousness_metrics_essential_or_not__cs_mean__cs_mean.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot(111)
    ax=dplot.plot(linestyle='',marker='o',ax=ax,**params['plot'])
    dplot.apply(lambda x : ax.text(x['x']+0.05,x['ROC AUC mean']-0.05,x[params['coltext']]), axis=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(dplot['classifier'],ha='left')
    ax.set_ylim(params['ylim'])
    ax.set_xlabel('classifier')
    ax.set_ylabel('ROC AUC')
    ax.get_legend().remove()
    return ax
# Figure EV3
## panel A
### plot#1
def plot_dist___of_interactions_paralog_or_singleton(plotp="plot/dist___of_interactions_paralog_or_singleton",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist___of_interactions_paralog_or_singleton.tsv");
    params_saved=yaml.load(open(f"plot/dist___of_interactions_paralog_or_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params)
    return ax
## panel B
### plot#1
def plot_dist_fpkm__log2_scale__paralog_or_singleton(plotp="plot/dist_fpkm__log2_scale__paralog_or_singleton",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_fpkm__log2_scale__paralog_or_singleton.tsv");
    params_saved=yaml.load(open(f"plot/dist_fpkm__log2_scale__paralog_or_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params)
    return ax
## panel C
### plot#1
def plot_scatter_fpkm__log2_scale__paralog_or_singleton_paralog_singleton(plotp="plot/scatter_fpkm__log2_scale__paralog_or_singleton_paralog_singleton",dplot=None,ax=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_fpkm__log2_scale__paralog_or_singleton_paralog_singleton.tsv");
    params_saved=yaml.load(open(f"plot/scatter_fpkm__log2_scale__paralog_or_singleton_paralog_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    from rohan.dandage.plot.scatter import plot_scatter
    ax=plot_scatter(dplot,params,ax=ax)
    return ax
## panel D
### plot#1
def plot_swarm_feature_importances_pcorr_paralog_or_singleton(plotp="plot/swarm_feature_importances_pcorr_paralog_or_singleton",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/swarm_feature_importances_pcorr_paralog_or_singleton.tsv");
    params_saved=yaml.load(open(f"plot/swarm_feature_importances_pcorr_paralog_or_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    ax=sns.swarmplot(data=dplot,dodge=False,
    size=10,ax=ax,**params['swarmplot'])
    ax.legend(bbox_to_anchor=[1,1],frameon=True)
    return ax
## panel E
### plot#1
def plot_heatmap_dists_cs_fpkm__log2_scale__mean_paralog_or_singleton(plotp="plot/heatmap_dists_cs_fpkm__log2_scale__mean_paralog_or_singleton",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/heatmap_dists_cs_fpkm__log2_scale__mean_paralog_or_singleton.tsv");
    params_saved=yaml.load(open(f"plot/heatmap_dists_cs_fpkm__log2_scale__mean_paralog_or_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.annot import pval2annot,annot_heatmap
    from rohan.dandage.plot.colors import get_cmap_subset
    if ax is None:ax=plt.subplot()
    ax=sns.heatmap(dplot.pivot_table(columns=params['colcolumns'],index=params['colindex'],
    values=params['colvalues'],),
    cmap=get_cmap_subset('Reds',0.1,0.7),
    cbar_kws={'label':'CS median'},
    ax=ax)
    annot_heatmap(ax, dannot=dplot.fillna(0).pivot_table(index='gene subset (dataset)',columns=['FPKM (log2 scale) mean'],
    values=params['colvaluesannot']).replace(0,np.nan).T.applymap(lambda x : pval2annot(x,fmt='<',alternative='two-sided')),
    xoff=0, yoff=0, kws_text={'color':'k','va':'center'}, annot_left='(', annot_right=')', annothalf='upper')
    return ax
# Figure 5
## panel A
### plot#1
def plot_contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotnone(plotp="plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotnone",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotnone.tsv");
    params_saved=yaml.load(open(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotnone.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.contour import plot_contourf,annot_contourf
    from rohan.dandage.plot.colors import get_cmap_subset
    fig,ax=plot_contourf(
    x=dplot[params['colx']].values,
    y=dplot[params['coly']].values,
    z=dplot[params['colz']].values,
    grid_n=25,
    params_contourf={'cmap':get_cmap_subset('binary_r',0,0.8)},
    labelx=params['colx'],labely=params['coly'],labelz=params['colz'],
    ax=ax,fig=fig,
    #             figsize=[3,3],
    test=False)
    fig,ax=annot_contourf(params['colx'],params['coly'],params['colz'],dplot,params['annot'],
    ax=ax,fig=fig)
    ax.set_xlim(-0.5,dplot[params['colx']].max())
    ax.set_ylim(-0.5,dplot[params['coly']].max())
    return ax
## panel B
### plot#1
def plot_contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotparalog_or_singleton(plotp="plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotparalog_or_singleton",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotparalog_or_singleton.tsv");
    params_saved=yaml.load(open(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotparalog_or_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.contour import plot_contourf,annot_contourf
    from rohan.dandage.plot.colors import get_cmap_subset
    fig,ax=plot_contourf(
    x=dplot[params['colx']].values,
    y=dplot[params['coly']].values,
    z=dplot[params['colz']].values,
    grid_n=25,
    params_contourf={'cmap':get_cmap_subset('binary_r',0,0.8)},
    labelx=params['colx'],labely=params['coly'],labelz=params['colz'],
    ax=ax,fig=fig,
    #             figsize=[3,3],
    test=False)
    fig,ax=annot_contourf(params['colx'],params['coly'],params['colz'],dplot,params['annot'],
    ax=ax,fig=fig)
    ax.set_xlim(-0.5,dplot[params['colx']].max())
    ax.set_ylim(-0.5,dplot[params['coly']].max())
    return ax
## panel C
### plot#1
def plot_contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotheteromer_or_not__all_ppi_(plotp="plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotheteromer_or_not__all_ppi_",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotheteromer_or_not__all_ppi_.tsv");
    params_saved=yaml.load(open(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypeall_annotheteromer_or_not__all_ppi_.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.contour import plot_contourf,annot_contourf
    from rohan.dandage.plot.colors import get_cmap_subset
    fig,ax=plot_contourf(
    x=dplot[params['colx']].values,
    y=dplot[params['coly']].values,
    z=dplot[params['colz']].values,
    grid_n=25,
    params_contourf={'cmap':get_cmap_subset('binary_r',0,0.8)},
    labelx=params['colx'],labely=params['coly'],labelz=params['colz'],
    ax=ax,fig=fig,
    #             figsize=[3,3],
    test=False)
    fig,ax=annot_contourf(params['colx'],params['coly'],params['colz'],dplot,params['annot'],
    ax=ax,fig=fig)
    ax.set_xlim(-0.5,dplot[params['colx']].max())
    ax.set_ylim(-0.5,dplot[params['coly']].max())
    return ax
# Figure EV4
## panel A
### plot#1
def plot_contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotnone(plotp="plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotnone",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotnone.tsv");
    params_saved=yaml.load(open(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotnone.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.contour import plot_contourf,annot_contourf
    from rohan.dandage.plot.colors import get_cmap_subset
    fig,ax=plot_contourf(
    x=dplot[params['colx']].values,
    y=dplot[params['coly']].values,
    z=dplot[params['colz']].values,
    grid_n=25,
    params_contourf={'cmap':get_cmap_subset('binary_r',0,0.8)},
    labelx=params['colx'],labely=params['coly'],labelz=params['colz'],
    ax=ax,fig=fig,
    #             figsize=[3,3],
    test=False)
    fig,ax=annot_contourf(params['colx'],params['coly'],params['colz'],dplot,params['annot'],
    ax=ax,fig=fig)
    ax.set_xlim(-0.5,dplot[params['colx']].max())
    ax.set_ylim(-0.5,dplot[params['coly']].max())
    return ax
## panel B
### plot#1
def plot_contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotparalog_or_singleton(plotp="plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotparalog_or_singleton",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotparalog_or_singleton.tsv");
    params_saved=yaml.load(open(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotparalog_or_singleton.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.contour import plot_contourf,annot_contourf
    from rohan.dandage.plot.colors import get_cmap_subset
    fig,ax=plot_contourf(
    x=dplot[params['colx']].values,
    y=dplot[params['coly']].values,
    z=dplot[params['colz']].values,
    grid_n=25,
    params_contourf={'cmap':get_cmap_subset('binary_r',0,0.8)},
    labelx=params['colx'],labely=params['coly'],labelz=params['colz'],
    ax=ax,fig=fig,
    #             figsize=[3,3],
    test=False)
    fig,ax=annot_contourf(params['colx'],params['coly'],params['colz'],dplot,params['annot'],
    ax=ax,fig=fig)
    ax.set_xlim(-0.5,dplot[params['colx']].max())
    ax.set_ylim(-0.5,dplot[params['coly']].max())
    return ax
## panel C
### plot#1
def plot_contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotheteromer_or_not__all_ppi_(plotp="plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotheteromer_or_not__all_ppi_",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotheteromer_or_not__all_ppi_.tsv");
    params_saved=yaml.load(open(f"plot/contour_fpkm__log2_scale____of_interactions__log2_scale__cs_mean_ppitypedirect_annotheteromer_or_not__all_ppi_.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.contour import plot_contourf,annot_contourf
    from rohan.dandage.plot.colors import get_cmap_subset
    fig,ax=plot_contourf(
    x=dplot[params['colx']].values,
    y=dplot[params['coly']].values,
    z=dplot[params['colz']].values,
    grid_n=25,
    params_contourf={'cmap':get_cmap_subset('binary_r',0,0.8)},
    labelx=params['colx'],labely=params['coly'],labelz=params['colz'],
    ax=ax,fig=fig,
    #             figsize=[3,3],
    test=False)
    fig,ax=annot_contourf(params['colx'],params['coly'],params['colz'],dplot,params['annot'],
    ax=ax,fig=fig)
    ax.set_xlim(-0.5,dplot[params['colx']].max())
    ax.set_ylim(-0.5,dplot[params['coly']].max())
    return ax
# Figure 6
## panel A
### plot#1
def plot_schem_asymmetry(plotp="plot/schem_asymmetry",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/schem_asymmetry.tsv");
    params_saved=yaml.load(open(f"plot/schem_asymmetry.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    X = np.array([
    [0,0.75], [1,0.95], [1, 0.75], 
    [0,0.5], [1,0.70], [1, 0.5], 
    [0,0.25],[0, 0.45], [1,0.45], [1, 0.25], 
    [0,0],   [0, 0.20], [1,0.20], [1, 0], 
    ])
    Y = [
    'black', 'black', 'black', 
    'gray', 'gray', 'gray',
    'black', 'black', 'black','gray', 
    'gray', 'gray', 'gray','gray',
    ]
    if ax is None:ax=plt.subplot()
    t1 = plt.Polygon(X[:3,:], color=Y[0])
    ax.add_patch(t1)
    t2 = plt.Polygon(X[3:6,:], color=Y[3])
    ax.add_patch(t2)
    t1 = plt.Polygon(X[6:10,:], color=Y[0])
    ax.add_patch(t1)
    t2 = plt.Polygon(X[10:14,:], color=Y[3])
    ax.add_patch(t2)
    ax.scatter(X[:, 0], X[:, 1], s = 0, color = Y[:])
    _=ax.text(0,1,'P1')
    _=ax.text(1,1,'P2',ha='right')
    _=[ax.text(1.1,y,s) for y,s in zip(np.arange(0.05,1,0.25),['deleteriousness\nupon LOF','mRNA\nexpression','deleteriousness\nupon LOF','mRNA\nexpression',])]
    ax.annotate(r"$\{$",fontsize=80,
    xy=(-0.55, 0.57), weight='ultralight',
    color='lightgray'
    )
    _=ax.text(-0.7,0.60,'Asymmetrical\nexpression')
    ax.annotate(r"$\{$",fontsize=80,
    xy=(-0.55, 0.07), weight='ultralight',
    color='lightgray'
    )
    _=ax.text(-0.7,0.07,'Symmetrical\nexpression')
    ax.set_xlim(-0.8,1.5)
    ax.set_ylim(0,1.05)
    ax.set_axis_off()
    return ax
## panel B
### plot#1
def plot_dist_heteromer_or_not__all_ppi__cs_mean_paralog_p1_mrna_expression____p2_mrna_expression__heteromer_or_not__all_ppi_(plotp="plot/dist_heteromer_or_not__all_ppi__cs_mean_paralog_p1_mrna_expression____p2_mrna_expression__heteromer_or_not__all_ppi_",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_heteromer_or_not__all_ppi__cs_mean_paralog_p1_mrna_expression____p2_mrna_expression__heteromer_or_not__all_ppi_.tsv");
    params_saved=yaml.load(open(f"plot/dist_heteromer_or_not__all_ppi__cs_mean_paralog_p1_mrna_expression____p2_mrna_expression__heteromer_or_not__all_ppi_.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    params_ax={
    #     'ylim':[-0.5,1],
    }        
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params,params_ax=params_ax)
    ax.set_xlabel('\n')
    return ax
## panel C
### plot#1
def plot_dist_assymetry_heteromer_not_heteromer(plotp="plot/dist_assymetry_heteromer_not_heteromer",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_assymetry_heteromer_not_heteromer.tsv");
    params_saved=yaml.load(open(f"plot/dist_assymetry_heteromer_not_heteromer.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    dplot=dplot.set_index(['gene ids','cell line'])
    if ax is None:ax=plt.subplot()
    [sns.distplot(dplot[subset].dropna(),hist=False,color=params['subset2color'][subset],ax=ax,label=subset,norm_hist=True) for subset in dplot]
    ax.set_xlabel('asymmetry in mRNA expression (P1-P2)/(P1+P2)')
    ax.set_ylabel('density')
    ax.set_xlim(0,1)
    ax.legend()
    return ax
## panel D
### plot#1
def plot_plot_assymetry_cs2_1_all_ppi_heteromer(plotp="plot/plot_assymetry_cs2_1_all_ppi_heteromer",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/plot_assymetry_cs2_1_all_ppi_heteromer.tsv");
    params_saved=yaml.load(open(f"plot/plot_assymetry_cs2_1_all_ppi_heteromer.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    dplot.plot(ax=ax,**params['plot'])
    ax.set_xticklabels([""]+dplot['FPKM (P1-P2)/(P1+P2) bin'].tolist())
    ax.set_xlabel('FPKM\n(P1-P2)/(P1+P2)\nequal size bins')
    ax.set_ylabel('CS mean\nP1-P2')
    return ax
### plot#2
def plot_plot_assymetry_cs2_1_all_ppi_not_heteromer(plotp="plot/plot_assymetry_cs2_1_all_ppi_not_heteromer",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/plot_assymetry_cs2_1_all_ppi_not_heteromer.tsv");
    params_saved=yaml.load(open(f"plot/plot_assymetry_cs2_1_all_ppi_not_heteromer.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    dplot.plot(ax=ax,**params['plot'])
    ax.set_xticklabels([""]+dplot['FPKM (P1-P2)/(P1+P2) bin'].tolist())
    ax.set_xlabel('FPKM\n(P1-P2)/(P1+P2)\nequal size bins')
    ax.set_ylabel('CS mean\nP1-P2')
    return ax
### plot#3
def plot_plot_assymetry_cs2_all_ppi_heteromer(plotp="plot/plot_assymetry_cs2_all_ppi_heteromer",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/plot_assymetry_cs2_all_ppi_heteromer.tsv");
    params_saved=yaml.load(open(f"plot/plot_assymetry_cs2_all_ppi_heteromer.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    dplot.plot(ax=ax,**params['plot'])
    ax.set_xticklabels([""]+dplot['FPKM (P1-P2)/(P1+P2) bin'].tolist())
    ax.set_xlabel('FPKM\n(P1-P2)/(P1+P2)\nequal size bins')
    ax.set_ylabel('CS mean\nP1-P2')
    return ax
### plot#4
def plot_plot_assymetry_cs2_all_ppi_not_heteromer(plotp="plot/plot_assymetry_cs2_all_ppi_not_heteromer",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/plot_assymetry_cs2_all_ppi_not_heteromer.tsv");
    params_saved=yaml.load(open(f"plot/plot_assymetry_cs2_all_ppi_not_heteromer.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    dplot.plot(ax=ax,**params['plot'])
    ax.set_xticklabels([""]+dplot['FPKM (P1-P2)/(P1+P2) bin'].tolist())
    ax.set_xlabel('FPKM\n(P1-P2)/(P1+P2)\nequal size bins')
    ax.set_ylabel('CS mean\nP1-P2')
    return ax
## panel E
### plot#1
def plot_scatter_cs_mean__cs1____of_interface_residues(plotp="plot/scatter_cs_mean__cs1____of_interface_residues",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_cs_mean__cs1____of_interface_residues.tsv");
    params_saved=yaml.load(open(f"plot/scatter_cs_mean__cs1____of_interface_residues.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    corrpval=sc.stats.spearmanr(dplot[params['colx']], dplot[params['coly']])
    label=f"{params['coly']}\n($\\rho$={corrpval[0]:.1f} {pval2annot(corrpval[1],fmt='<',linebreak=False)})"
    ax=sns.regplot(data=dplot,x=params['colx'],y=params['coly'],
    scatter_kws={'alpha':0.05,'color':params['color']},
    line_kws={'label':label,'color':params['color']},
    ax=ax,)
    ax.set_ylabel(params['coly'].split(' (')[0])
    ax.legend(bbox_to_anchor=[1,1])
    return ax
### plot#2
def plot_scatter_cs_mean__cs2____of_interface_residues(plotp="plot/scatter_cs_mean__cs2____of_interface_residues",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_cs_mean__cs2____of_interface_residues.tsv");
    params_saved=yaml.load(open(f"plot/scatter_cs_mean__cs2____of_interface_residues.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    corrpval=sc.stats.spearmanr(dplot[params['colx']], dplot[params['coly']])
    label=f"{params['coly']}\n($\\rho$={corrpval[0]:.1f} {pval2annot(corrpval[1],fmt='<',linebreak=False)})"
    ax=sns.regplot(data=dplot,x=params['colx'],y=params['coly'],
    scatter_kws={'alpha':0.05,'color':params['color']},
    line_kws={'label':label,'color':params['color']},
    ax=ax,)
    ax.set_ylabel(params['coly'].split(' (')[0])
    ax.legend(bbox_to_anchor=[1,1])
    return ax
### plot#3
def plot_scatter_cs_mean__cs2_1____of_interface_residues(plotp="plot/scatter_cs_mean__cs2_1____of_interface_residues",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_cs_mean__cs2_1____of_interface_residues.tsv");
    params_saved=yaml.load(open(f"plot/scatter_cs_mean__cs2_1____of_interface_residues.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    corrpval=sc.stats.spearmanr(dplot[params['colx']], dplot[params['coly']])
    label=f"{params['coly']}\n($\\rho$={corrpval[0]:.1f} {pval2annot(corrpval[1],fmt='<',linebreak=False)})"
    ax=sns.regplot(data=dplot,x=params['colx'],y=params['coly'],
    scatter_kws={'alpha':0.05,'color':params['color']},
    line_kws={'label':label,'color':params['color']},
    ax=ax,)
    ax.set_ylabel(params['coly'].split(' (')[0])
    ax.legend(bbox_to_anchor=[1,1])
    return ax
### plot#4
def plot_scatter_cs_mean__cs3____of_interface_residues(plotp="plot/scatter_cs_mean__cs3____of_interface_residues",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/scatter_cs_mean__cs3____of_interface_residues.tsv");
    params_saved=yaml.load(open(f"plot/scatter_cs_mean__cs3____of_interface_residues.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    if ax is None:ax=plt.subplot()
    corrpval=sc.stats.spearmanr(dplot[params['colx']], dplot[params['coly']])
    label=f"{params['coly']}\n($\\rho$={corrpval[0]:.1f} {pval2annot(corrpval[1],fmt='<',linebreak=False)})"
    ax=sns.regplot(data=dplot,x=params['colx'],y=params['coly'],
    scatter_kws={'alpha':0.05,'color':params['color']},
    line_kws={'label':label,'color':params['color']},
    ax=ax,)
    ax.set_ylabel(params['coly'].split(' (')[0])
    ax.legend(bbox_to_anchor=[1,1])
    return ax
# Figure EV5
## panel A
### plot#1
def plot_plot_asymmetry_bins_fraction_cs_cs2_1(plotp="plot/plot_asymmetry_bins_fraction_cs_cs2_1",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/plot_asymmetry_bins_fraction_cs_cs2_1.tsv");
    params_saved=yaml.load(open(f"plot/plot_asymmetry_bins_fraction_cs_cs2_1.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.colors import get_cmap_subset
    from rohan.dandage.plot.ax_ import set_colorbar
    if ax is None:ax=plt.subplot()
    ax_pc=ax.scatter(x=dplot.index,y=dplot[params['coly']],c=dplot[params['colc']],
    cmap=get_cmap_subset(plt.get_cmap('Reds'), 0.2, 0.8))
    fig=set_colorbar(fig,ax,ax_pc,label='P1(CS)-P2(CS)',
    bbox_to_anchor=(1.01, -1.0, 0.5, 2))
    _=ax.set_xticks(np.arange(min(dplot.index), max(dplot.index)+1, 1.0))
    _=ax.set_xticklabels(dplot[params['colxorder']],rotation=90)
    ax.set_xlabel('asymmetry in mRNA expression (P1-P2)/(P1+P2)')
    ax.set_ylabel(params['coly'])
    ax.axhline(y=0.5, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],linestyle='--',color='gray')
    return ax
## panel B
### plot#1
def plot_dist_assymetry_corr_heteromer_or_not__all_ppi__cs2_1(plotp="plot/dist_assymetry_corr_heteromer_or_not__all_ppi__cs2_1",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_assymetry_corr_heteromer_or_not__all_ppi__cs2_1.tsv");
    params_saved=yaml.load(open(f"plot/dist_assymetry_corr_heteromer_or_not__all_ppi__cs2_1.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params,)
    return ax
## panel C
### plot#1
def plot_dist_paralog_expression_corr_p1_rna_expression__p2_rna_expression___regular_correlation_(plotp="plot/dist_paralog_expression_corr_p1_rna_expression__p2_rna_expression___regular_correlation_",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_paralog_expression_corr_p1_rna_expression__p2_rna_expression___regular_correlation_.tsv");
    params_saved=yaml.load(open(f"plot/dist_paralog_expression_corr_p1_rna_expression__p2_rna_expression___regular_correlation_.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params,)
    ax.set_xlabel('')
    return ax
## panel D
### plot#1
def plot_dist_paralog_expression_corr_p1_protein_expression__p2_protein_expression___partial_correlation_(plotp="plot/dist_paralog_expression_corr_p1_protein_expression__p2_protein_expression___partial_correlation_",dplot=None,ax=None,fig=None,params=None):
    if dplot is None:dplot=read_table(f"plot/dist_paralog_expression_corr_p1_protein_expression__p2_protein_expression___partial_correlation_.tsv");
    params_saved=yaml.load(open(f"plot/dist_paralog_expression_corr_p1_protein_expression__p2_protein_expression___partial_correlation_.yml","r"));params=params_saved if params is None else {k:params[k] if k in params else params_saved[k] for k in params_saved};
    from rohan.dandage.plot.dist import plot_dist_comparison
    if ax is None:ax=plt.subplot(111)
    ax=plot_dist_comparison(dplot,ax=ax,**params,)
    ax.set_xlabel('')
    return ax
