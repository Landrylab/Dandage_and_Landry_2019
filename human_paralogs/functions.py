from rohan.dandage.db.ensembl import *
def get_dgene(test=False):
    """
    Get gene annotations.
    :params test: verbose.

    """

    from human_paralogs.global_vars import cfg

    dn2df={}
    for dn, en in zip(['',' (release75)'],[cfg['ensembl'],cfg['ensembl75']]):
        id2name={}
        for g in en.genes():
            if g.is_protein_coding:
                id2name[g.id]=g.name
        df=pd.DataFrame(pd.Series(id2name,name=f'gene name{dn}'))
        dn2df[dn]=df

    dgene=pd.concat(dn2df,axis=1,sort=True).reset_index()
    dgene.columns=[t[1] for t in tuple(dgene.columns)]
    dgene=dgene.rename(columns={'':'gene id'})

    if test:
        print(dgene.shape)
    dgene=dgene.drop_duplicates(subset=['gene name','gene name (release75)'])

    if test:
        print(dgene.shape)

    to_table(dgene,cfg['dgenep'])

    if test:
        return dgenefrom

def get_dcslin():
    """
    Get CS data.

    """
    dcsmap=read_table(dcsmapp)

    dcslin=dmap2lin(dcsmap,idxn='gene name', coln='dataset cell line', colvalue_name='CS')

    from rohan.dandage.io_strs import get_prefix,get_suffix
    dcslin['dataset']=dcslin['dataset cell line'].apply(lambda x : get_prefix(x,' '))
    print(dcslin['dataset'].unique())

    dcslin['cell line']=dcslin['dataset cell line'].apply(lambda x : get_suffix(x,' '))
    dcslin['cell line'].unique().shape

    print(dcslin['dataset'].isnull().sum(),dcslin['cell line'].isnull().sum())

    to_table(dcslin,'data_cs/dcslin.pqt')

def get_dcsagg():
    """
    Get CS data aggregated.

    """

    dcslin=read_table(dcslinp)

    dcsagg=dcslin.groupby(['dataset','gene name']).agg({'CS':[np.mean,np.median,np.std]}).reset_index()
    dcsagg.columns=coltuples2str(dcsagg.columns)
    dataset2gn=dcsagg.groupby('dataset').agg({'gene name':list}).to_dict()['gene name']

    from rohan.dandage.io_sets import intersections
    print([len(dataset2gn[k]) for k in dataset2gn])
    print(intersections(dataset2gn))

    dcsagg=dcsagg.pivot_table(columns='dataset',index='gene name',values=['CS mean','CS median','CS std'])
    dcsagg.columns=coltuples2str(dcsagg.columns)

    print(dcsagg.head())

    to_table(dcsagg,dcsaggp)from human_paralogs.global_vars import *

def get_dexpression_lin():
    """
    Get FPKM data.

    """

    dgenes=read_table(cfg['dgenep'])
    dexpression_=read_table(dexpression_rawp).reset_index()

    dexpression_=dexpression_.rename(columns={'gene_short_name':'gene name'})
    dexpression_=dexpression_.loc[dexpression_['gene name'].isin(dgenes['gene name']),:]
    print(dexpression_['gene name'].unique().shape)

    dexpression_=dexpression_.loc[(dexpression_['FPKM_status']=='OK'),['FPKM','gene name','cell line']]
    dexpression_=dflogcol(dexpression_,'FPKM',base=2,pcount=1)
    dexpression_=dexpression_.loc[(~dexpression_['gene name'].isin([s for s in dexpression_['gene name'] if ',' in s])),:]

    dexpression_['gdc id']=dexpression_['cell line'].apply(lambda x : x.split('.')[0])

    dceline=read_table(dcelinep)
    dexpression_=dexpression_.merge(dceline.loc[:,['Broad_ID','gdc id','cell line crispr']],
                                          on='gdc id',how='left')
    print(dexpression_['Broad_ID'].unique().shape)

    dexpression_=dexpression_.dropna(subset=['Broad_ID'])
    dexpression_=dexpression_.loc[(dexpression_['gene name']!='-'),:]
    print(dexpression_['Broad_ID'].unique().shape)

    dexpression_['cell line']=dexpression_['Broad_ID']
    dexpression_=dexpression_.loc[:,['gene name', 'cell line', 'FPKM (log2 scale)']]
    dexpression_=dexpression_.groupby(['gene name','cell line']).agg({'FPKM (log2 scale)':np.mean}).reset_index()

    to_table(dexpression_,cfg['dexpression_linp'])

def get_dexpression_agg():
    """
    Get FPKM data aggregated.

    """

    dgenes=read_table(cfg['dgenep'])
    dexpression_=read_table(cfg['dexpression_linp'])

    from rohan.dandage.io_dfs import coltuples2str
    dexpression_agg=dexpression_.groupby('gene name').agg({'FPKM':[np.median,np.mean,np.std],
                                                     'FPKM (log2 scale)':[np.median,np.mean,np.std]})
    dexpression_agg.columns=coltuples2str(dexpression_agg.columns)

    to_table(dexpression_agg,cfg['dexpression_aggp'])

from human_paralogs.global_vars import *

def get_dgo_raw_enrichr(gene_sets=['GO_Molecular_Function_2018',
                                    'GO_Biological_Process_2018',
                                    'GO_Cellular_Component_2018',
                                           ],
                        test=False,
                       ):
    """
    Get enrichment.

    """

    if isinstance(gene_sets,dict):
        gene_sets_names=list(gene_sets.keys())
    else:
        gene_sets_names=gene_sets

    import gseapy as gp
    dgene_annot=read_table(cfg['dgene_annotp'])

    from rohan.dandage.io_strs import make_pathable_string
    for col in [k for k in cfg['colgene_subset2classes'] if k.endswith('PPI)') or k.startswith('paralog or')]:
        for subsetn in cfg['colgene_subset2classes'][col][:-1]:
            for gene_set_name in gene_sets_names:
                outp=make_pathable_string(f"data_go/dgo_raw_enrichr/{col}/{subsetn}/{gene_set_name}")
                if test:
                    print(outp)
                if not exists(outp) or test:
                    enr = gp.enrichr(gene_list=dgene_annot.loc[(dgene_annot[col]==subsetn),'gene name'].tolist(),
                                 # or gene_list=glist
                                 description=f"{make_pathable_string(col)}_{subsetn}",
                                 gene_sets=[gene_set_name] if isinstance(gene_sets,list) else gene_sets[gene_set_name],
                                 outdir=outp,
                                 background=str(dgene_annot['paralog or singleton'].value_counts()['paralog']),
                                 cutoff=0.05 ,
                                 no_plot=True, verbose=True if test else False,
                                )
                                
def annot_paralogs_essential(din,genes_ess,genes_ness,test=False):
    """
    Subsets of paralogs based on essentiality.

    """

    print(len(genes_ess),len(genes_ness))

    for genei in [1,2]:
        din.loc[din[f'gene{genei} name'].isin(genes_ess),f'essential or non-essential gene{genei}']='essential'
        din.loc[din[f'gene{genei} name'].isin(genes_ness),f'essential or non-essential gene{genei}']='non-essential'

    def classify(x):
        clss=unique_dropna([x['essential or non-essential gene1'],x['essential or non-essential gene2']])
        if len(clss)==2:
            return 'both'
        elif 'non-essential' in clss:
            return 'non-essential'
        elif 'essential' in clss:
            return 'essential'
        else:
            return np.nan

    if test:
        print(din['essential or non-essential gene1'].value_counts())
        print(din['essential or non-essential gene2'].value_counts())
        print(din.apply(lambda x :  classify(x),axis=1).value_counts())

    return din.apply(lambda x :  classify(x),axis=1)

def get_dgene_paralog_or_singleton(test=False):
    """
    Subsets of paralogs.

    """

    from human_paralogs.global_vars import cfg
    
    dsingletons=read_table(cfg['dgene_singletonsp'])
    dparalogs=read_table(cfg['dparalogp'])

    genes_paralogs=list(np.unique(dparalogs['gene1 name'].tolist()+dparalogs['gene2 name'].tolist()))
    genes_singletons=list(set(dsingletons['gene name'].tolist()).difference(genes_paralogs))
    genename2genesubset=dict(zip(genes_paralogs+genes_singletons,
                                 ['paralog' for i in genes_paralogs]+['singleton' for i in genes_singletons]))

    dgene_paralog_or_singleton=pd.DataFrame(pd.Series(genename2genesubset)).reset_index()

    dgene_paralog_or_singleton.columns=['gene name','paralog or singleton']

    to_table(dgene_paralog_or_singleton,cfg['dgene_paralog_or_singletonp'])

    if test:
        return dgene_paralog_or_singleton
    
def get_dgene_essetial(test=False):
    """
    Essential genes.

    """

    from human_paralogs.global_vars import cfg

    dcsagg=read_table(cfg['dcsaggp'])

    dataset2genes={}
    
    dgene=read_table(cfg['dgenep']).loc[:,['gene name']]
    dataset2genes['bagel']={'essential':read_table(f"{cfg['bagelsrcp']}/CEGv2.txt",
                                                   params_read_csv={'sep':'\t','names':['gene name','Unnamed','Unnamed2']})['gene name'].unique().tolist(),

                            'non-essential':read_table(f"{cfg['bagelsrcp']}/NEGv1.txt",
                                                       params_read_csv={'sep':'\t','names':['gene name','Unnamed','Unnamed2']})['gene name'].unique().tolist(),

                           }

    dataset2genes['depmap']={'essential':pd.read_csv(f"{cfg['depmapp']}/essential_genes.txt",
                                                     sep='\t')['gene'].apply(lambda x :x.split(' ')[0]).unique().tolist(),

                            'non-essential':pd.read_csv(f"{cfg['depmapp']}/nonessential_genes.txt",
                                                        sep='\t')['gene'].apply(lambda x :x.split(' ')[0]).unique().tolist(),

                           }

    dataset2genes['depmap+bagel']={
    'essential':np.unique(dataset2genes['depmap']['essential']+dataset2genes['bagel']['essential']),
    'non-essential':np.unique(dataset2genes['depmap']['non-essential']+dataset2genes['bagel']['non-essential']),
              }

    for dataset in dataset2genes:
        for gene_subset in dataset2genes[dataset]:
            dgene.loc[(dgene['gene name'].isin(dataset2genes[dataset][gene_subset])),f'essential or non-essential ({dataset})']=gene_subset
        dgene.loc[pd.isnull(dgene[f'essential or non-essential ({dataset})']),f'essential or non-essential ({dataset})']='unclassified'

    to_table(dgene,cfg['dgene_essetialp'])

    if test:
        return dgene

def get_dgene_annot(test=False):
    """
    Get gene annotations.

    """
    
    from human_paralogs.global_vars import cfg
    
    dn2df={}
    dn2df['dgene_paralog_or_singleton']=read_table(cfg['dgene_paralog_or_singletonp'])
    dn2df['dint_degnonselfmap']=read_table(cfg['dint_degnonselfmapp'])

    dn2df['dint_degnonselfmap']=dn2df['dint_degnonselfmap'].loc[:,['gene name']+[c for c in dn2df['dint_degnonselfmap'] if '(log2 scale)' in c ]]

    dn2df['dint_degnonselfmap']['# of interactions (all PPI) (log2 scale)']=dn2df['dint_degnonselfmap'].loc[:,['# of interactions (all biogrid) (log2 scale)','# of interactions (all intact) (log2 scale)']].T.mean()

    dn2df['dint_degnonselfmap']['# of interactions (direct PPI) (log2 scale)']=dn2df['dint_degnonselfmap'].loc[:,['# of interactions (direct biogrid) (log2 scale)','# of interactions (direct intact) (log2 scale)']].T.mean()

    dn2df['dhomomers']=read_table(cfg['dgene_homomersp'])
    dn2df['dheteromers']=read_table(cfg['dparalog_heteromerslinp'])
    dn2df['essential']=read_table(cfg['dgene_essetialp']).drop_duplicates(subset=['gene name'],keep=False)
    
    dgene_annot=read_table(cfg['dgenep'])
    if test:
        print(dgene_annot.shape)
    dgene_annot=dgene_annot.merge(dn2df['dgene_paralog_or_singleton'],how='left',on='gene name')

    if test:
        print(dgene_annot.shape)
    dgene_annot=dgene_annot.merge(dn2df['dint_degnonselfmap'],how='left',on='gene name')

    if test:
        print(dgene_annot.shape)
    dgene_annot=dgene_annot.merge(dn2df['dhomomers'],how='left',on='gene name')

    if test:
        print(dgene_annot.shape)
    dgene_annot=dgene_annot.merge(dn2df['dheteromers'],how='left',on='gene name')

    if test:
        print(dgene_annot.shape)
    dgene_annot=dgene_annot.merge(dn2df['essential'],how='left',on='gene name')

    if test:
        print(dgene_annot.shape)

    to_table(dgene_annot,cfg['dgene_annotp'])

    cols_classes=[c for c in dgene_annot.dtypes[dgene_annot.dtypes=='object'].index if (not 'gene' in c) and (not 'bin' in c) and (not 'age' in c)  and (not 'expPattern' in c)]

    dstats=dgene_annot.loc[:,cols_classes].apply(pd.Series.value_counts)
    dstats.index.name='gene subset'

    to_table(dstats,cfg['dgene_annot_statsp'])

    if test:
        return dstats

from rohan.global_imports import *

def get_dmerge_agg_annot(test=False):
    """
    Merge data annotations.

    """

    from human_paralogs.global_vars import cfg

    dgene_annot=read_table(cfg['dgene_annotp'])
    dgene_annot=dgene_annot.dropna(subset=['gene name','gene name (release75)']).drop_duplicates(subset=['gene name (release75)'])

    cols_classes=[c for c in dgene_annot.dtypes[dgene_annot.dtypes=='object'].index if (not 'gene' in c) and (not 'bin' in c) and (not 'age' in c)  and (not 'expPattern' in c)]

    params_annot={'how':'left','left_on':'gene name (release75)','right_on':'gene name','validate':'1:1'}
    dmerge_agg_annot=dgene_annot.merge(read_table(cfg['dexpression_aggp']).reset_index(),
                                              **params_annot).merge(read_table(cfg['dcsaggp']).reset_index(),
                                                                    **params_annot)

    for col in cols_classes:
        dmerge_agg_annot.loc[pd.isnull(dmerge_agg_annot[col]),col]='unclassified'

    print(dmerge_agg_annot.shape)

    to_table(dmerge_agg_annot,cfg['dmerge_agg_annotp'])

    if test:
        return dmerge_agg_annot

def get_dmerge_lin(test=False):
    """
    Merged data in linear format.

    """

    from human_paralogs.global_vars import cfg

    dcslin=read_table(cfg['dcslinp'])

    dmerge_lin=dcslin.merge(read_table(cfg['dexpression_linp']),on=['gene name','cell line'],how='outer',validate='m:1')

    if test:
        print(dmerge_lin.dropna()['cell line'].unique().shape)
        print(dmerge_lin['cell line'].unique().shape)
        print(dmerge_lin['dataset'].unique().shape)

    to_table(dmerge_lin,cfg['dmerge_linp'])

    if test:
        return dmerge_lin

def get_dmerge_lin_annot(test=False):
    """
    Annotate linear data.

    """

    from human_paralogs.global_vars import cfg
    from human_paralogs.curate03_annot import unclassify_dannot

    dmerge_lin=read_table(cfg['dmerge_linp'])
    dgene_annot=read_table(cfg['dgene_annotp'])

    dgene_annot=dgene_annot.dropna(subset=['gene name','gene name (release75)']).drop_duplicates(subset=['gene name (release75)'])
    dmerge_lin_annot=dmerge_lin.merge(dgene_annot.loc[:,['gene name','gene name (release75)']+list(cfg['colgene_subset2classes'].keys())],
                                      how='left',left_on='gene name',right_on='gene name (release75)',validate='m:1')
    dmerge_lin_annot=unclassify_dannot(dmerge_lin_annot)

    to_table(dmerge_lin_annot,cfg['dmerge_lin_annotp'])

    if test:
        return dmerge_lin_annot

def get_dmerge_agg_annot_paralog(test=False):
    """
    Merge paralog data.

    """

    from human_paralogs.global_vars import cfg
    from human_paralogs.curate03_annot import unclassify_dannot

    dparalog_annot=read_table(cfg['dparalog_annotp'])
    dparalog_annot=dparalog_annot.dropna(subset=['gene1 name','gene2 name','gene1 name (release75)','gene2 name (release75)']).drop_duplicates(subset=['gene1 name (release75)','gene2 name (release75)'])
    dmerge_agg_annot=read_table(cfg['dmerge_agg_annotp'])

    dmerge_agg_annot_paralog=merge_dfpairwithdf(dparalog_annot.loc[:,[c for c in dparalog_annot if not 'gene name gene' in c]],
                      dmerge_agg_annot.loc[:,cfg['dataset_cols']+['gene name']],
                      left_ons=['gene1 name', 'gene2 name'],
                      right_on='gene name', suffixes=[' gene1', ' gene2'])

    for cg1,cg2 in zip([f"{c} gene1" for c in cfg['dataset_cols']],[f"{c} gene2" for c in cfg['dataset_cols']]):
        dmerge_agg_annot_paralog[f'{cg1[:-6]} min']=dmerge_agg_annot_paralog.loc[:,[cg1,cg2]].T.min()
        dmerge_agg_annot_paralog[f'{cg1[:-6]} mean']=dmerge_agg_annot_paralog.loc[:,[cg1,cg2]].T.mean()
        dmerge_agg_annot_paralog[f'{cg1[:-6]} max']=dmerge_agg_annot_paralog.loc[:,[cg1,cg2]].T.max()
        dmerge_agg_annot_paralog[f'{cg1[:-6]} absolute delta']=dmerge_agg_annot_paralog.apply(lambda x : abs(x[cg1]-x[cg2]),axis=1)

    to_table(dmerge_agg_annot_paralog,cfg['dmerge_agg_annot_paralogp'])

    if test:
        return dmerge_agg_annot_paralog

def get_dmerge_lin_annot_paralog(test=False):
    """
    Merge paralog linear data.

    """

    from human_paralogs.global_vars import cfg

    dmerge_lin=read_table(cfg['dmerge_linp'])

    dmerge_lin.loc[pd.isnull(dmerge_lin['dataset']),'dataset']='FPKM'

    dparalog_annot=read_table(cfg['dparalog_annotp'])
    dparalog_annot=dparalog_annot.dropna(subset=['gene1 name','gene2 name','gene1 name (release75)','gene2 name (release75)']).drop_duplicates(subset=['gene1 name (release75)','gene2 name (release75)'])

    dmerge_lin_annot_paralog=merge_dfpairwithdf(dparalog_annot.loc[:,[c for c in dparalog_annot if not 'gene name gene' in c]],
                                   dmerge_lin,how='left',
                      left_ons=['gene1 name (release75)', 'gene2 name (release75)'],
                      right_on='gene name', right_ons_common=['dataset','cell line'],
                    suffixes=[' gene1', ' gene2'])

    for cg1,cg2 in zip([f"{c} gene1" for c in ['CS','FPKM (log2 scale)']],[f"{c} gene2" for c in ['CS','FPKM (log2 scale)']]):
        dmerge_lin_annot_paralog[f'{cg1[:-6]} min']=dmerge_lin_annot_paralog.loc[:,[cg1,cg2]].T.min()
        dmerge_lin_annot_paralog[f'{cg1[:-6]} mean']=dmerge_lin_annot_paralog.loc[:,[cg1,cg2]].T.mean()
        dmerge_lin_annot_paralog[f'{cg1[:-6]} max']=dmerge_lin_annot_paralog.loc[:,[cg1,cg2]].T.max()
        dmerge_lin_annot_paralog[f'{cg1[:-6]} absolute delta']=dmerge_lin_annot_paralog.apply(lambda x : abs(x[cg1]-x[cg2]),axis=1)
    dmerge_lin_annot_paralog=dmerge_lin_annot_paralog.dropna(subset=[c for c in dmerge_lin_annot_paralog if ('dataset' in c) or ('cell line' in c)],how='any')

    for col in list(cfg['colgene_subset2classes'].keys())+[c for c in dmerge_lin_annot_paralog if c.endswith(' sequence')]:
        if col in dmerge_lin_annot_paralog:
            dmerge_lin_annot_paralog[col]=dmerge_lin_annot_paralog[col].astype({col:str})

    dmerge_lin_annot_paralog=dmerge_lin_annot_paralog.loc[:,[c for c in dmerge_lin_annot_paralog if not c.endswith(' sequence')]]

    to_table(dmerge_lin_annot_paralog,cfg['dmerge_lin_annot_paralogp'])

    if test:
        return dmerge_lin_annot_paralog

def dmerge_lin_annot_paralog_sorted_expression(test=False):
    """
    Sort paralog data.

    """
    from human_paralogs.global_vars import cfg
    dmerge_lin_annot_paralog=read_table(cfg['dmerge_lin_annot_paralogp'])

    print(dmerge_lin_annot_paralog['cell line gene1'].shape,
    dmerge_lin_annot_paralog['dataset gene2'].unique(),

    dmerge_lin_annot_paralog.loc[(~pd.isnull(dmerge_lin_annot_paralog['FPKM (log2 scale) mean'])),'cell line gene1'].unique().shape)
    celllines=dmerge_lin_annot_paralog.loc[(~pd.isnull(dmerge_lin_annot_paralog['FPKM (log2 scale) mean'])),'cell line gene1'].unique().tolist()

    dmerge_lin_annot_paralog=dmerge_lin_annot_paralog.loc[(dmerge_lin_annot_paralog['cell line gene1'].isin(celllines)),:]
    df_=dmerge_lin_annot_paralog.apply(lambda x : (x['gene1 id'],x['gene2 id']) if x['FPKM (log2 scale) gene1']>x['FPKM (log2 scale) gene2'] else (x['gene2 id'],x['gene1 id']),axis=1).apply(pd.Series)

    df_.columns=['gene1 id high expression','gene2 id low expression']
    df_CS=dmerge_lin_annot_paralog.apply(lambda x : (x['CS gene1'],x['CS gene2']) if x['FPKM (log2 scale) gene1']>x['FPKM (log2 scale) gene2'] else (x['CS gene2'],x['CS gene1']),axis=1).apply(pd.Series)

    df_CS.columns=['gene1 CS high expression','gene2 CS low expression']
    df_FPKM=dmerge_lin_annot_paralog.apply(lambda x : (x['FPKM (log2 scale) gene1'],x['FPKM (log2 scale) gene2']) if x['FPKM (log2 scale) gene1']>x['FPKM (log2 scale) gene2'] else (x['FPKM (log2 scale) gene2'],x['FPKM (log2 scale) gene1']),axis=1).apply(pd.Series)
    df_FPKM.columns=['gene1 FPKM (log2 scale) high expression','gene2 FPKM (log2 scale) low expression']

    df=dmerge_lin_annot_paralog.join(df_).join(df_CS).join(df_FPKM)
    df=df.rename(columns={'cell line gene1':'cell line',
     'dataset gene1':'dataset'})

    to_table(df,cfg['dmerge_lin_annot_paralog_sorted_expressionp'])

    if test:
        return df

def get_dparalog_expression_merge():
    """
    Merge proteomics with transcriptomics

    """
    dparalog_expression_protein=read_table('data_expression_protein/dparalog_expression_protein.pqt')
    dparalog_expression_protein=dparalog_expression_protein.loc[(~pd.isnull(dparalog_expression_protein['cell line'])),:]
    dparalog_expression_rna=read_table(cfg['dmerge_lin_annot_paralog_sorted_expression_anap'])

    dparalog_expression_merge=dparalog_expression_rna.loc[:,[c for c in dparalog_expression_rna if c.startswith('gene') and len(c) < 20]+['cell line','FPKM (log2 scale) gene1','FPKM (log2 scale) gene2']].merge(

    dparalog_expression_protein.loc[:,[c for c in dparalog_expression_protein if c.startswith('gene') and c.endswith(' id')]+['cell line','protein abundance gene1 (log2 scale)','protein abundance gene2 (log2 scale)']],
    on=['gene1 id','gene2 id','cell line'],how='inner',
    )

    colds=[]
    for dataset in ['protein abundance','FPKM']:
        colds+=list(np.unique(dparalog_expression_merge.filter(like=dataset).columns.tolist()))

    print(colds)

    for c in colds:
        dparalog_expression_merge[c]=df2zscore(dparalog_expression_merge,c)

    to_table(dres,'data_merge/dparalog_expression_merge.tsv')

from rohan.global_imports import *

from human_paralogs.global_vars import cfg

def get_dparalog_annot():
    """
    Paralog annotations.

    """
    dparalog_annot=read_table(cfg['dparalog_annotp'])
    ppitypes=[' '.join(list(t)) for t in itertools.product(['all','direct','indirect'],['biogrid','intact'])]

    dn2df={}
    for ppitype in ppitypes:
        dparalog_annot.loc[ (dparalog_annot[f'homomer or not ({ppitype}) gene1']=='homomer'),'gene1 homomer']='P1P1'
        dparalog_annot.loc[~(dparalog_annot[f'homomer or not ({ppitype}) gene1']=='homomer'),'gene1 homomer']=''
        dparalog_annot.loc[ (dparalog_annot[f'homomer or not ({ppitype}) gene2']=='homomer'),'gene2 homomer']='P2P2'
        dparalog_annot.loc[~(dparalog_annot[f'homomer or not ({ppitype}) gene2']=='homomer'),'gene2 homomer']=''
        dparalog_annot.loc[ (dparalog_annot[f'heteromer or not ({ppitype})']=='heteromer'),'heteromer']='P1P2'
        dparalog_annot.loc[~(dparalog_annot[f'heteromer or not ({ppitype})']=='heteromer'),'heteromer']=''

        df=dparalog_annot.groupby(['gene1 homomer',
         'gene2 homomer',
         'heteromer']).agg({'gene names':len})
        df['PPI type']=ppitype
        dn2df[ppitype]=df
    df=pd.concat(dn2df,axis=0).reset_index()

    df=df.rename(columns={'gene names':'# of paralogs'}).set_index('PPI type').loc[:,['gene1 homomer','gene2 homomer','heteromer','# of paralogs']]

    to_table(df,cfg['dparalog_annot_statsp'])

from rohan.global_imports import *
from human_paralogs.global_vars import cfg
def get_paralog_corr(df,colds):
    """
    Get paralogs correlations.

    """
    from scipy.stats import spearmanr,pearsonr
    colgenesubsets=[c for c in cfg['colgene_subset2classes'] if c.startswith('heteromer') and 'PPI' in c]
    dres=pd.DataFrame(index=df['gene ids'].unique(),
                columns=['r pearson','p-val pearson','r spearman','p-val spearman'])

    for gene_idsi,gene_ids in enumerate(df['gene ids'].unique()):
        df_=df.loc[(df['gene ids']==gene_ids),['gene ids','cell line']+colds].drop_duplicates(subset=['cell line'])
        df_=df_.dropna(subset=colds)
        if len(df_)>len(df['cell line'].unique())*0.1:
            print(gene_idsi,end=' ')
            dres.loc[gene_ids,'r pearson'],dres.loc[gene_ids,'p-val pearson']=pearsonr(df_[colds[0]],df_[colds[1]])
            dres.loc[gene_ids,'r spearman'],dres.loc[gene_ids,'p-val spearman']=spearmanr(df_[colds[0]],df_[colds[1]])
    dres.index.name='gene ids'
    dres['r spearman']=dres['r spearman'].astype(float)
    dres['r pearson']=dres['r pearson'].astype(float)

    return dres

def get_dparalog_expression_corr():
    """
    Get paralogs expression correlations.

    """
    dparalog_expression_merge=read_table('data_merge/dparalog_expression_merge.tsv')

    dn2df={}

    dn2df['P1(protein expression):P2(protein expression) (partial correlation)']=get_paralog_pcorr(dparalog_expression_merge,colds).set_index('gene ids')

    dn2df['P1(protein expression):P2(protein expression) (regular correlation)']=get_paralog_corr(dparalog_expression_merge,colds[:2])

    dn2df['P1(RNA expression):P2(RNA expression) (regular correlation)']=get_paralog_corr(dparalog_expression_merge,colds[2:])

    dn2df['P1(protein expression):P1(RNA expression) (regular correlation)']=get_paralog_corr(dparalog_expression_merge,[colds[0],colds[2]])

    dn2df['P2(protein expression):P2(RNA expression) (regular correlation)']=get_paralog_corr(dparalog_expression_merge,[colds[1],colds[3]])

    dres=delunnamedcol(pd.concat(dn2df,axis=0,sort=True,names=['correlation type','gene ids']).rename(columns={'gene ids':'Unnamed'}).reset_index())

    to_table(dres,'data_mechanism/dparalog_expression_corr.tsv')