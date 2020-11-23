# import functions needed for the analysis
from rohan.global_imports import *
# paths to the files
import yaml
cfgp=f"{dirname(realpath(__file__))}/cfg.yml"
cfg=yaml.load(open(cfgp,'r'))
cfg={k:f"{dirname(realpath(__file__))}/{cfg[k]}" for k in cfg}
cfg['cfgp']=cfgp
## gene information from pyensembl objects
import pyensembl
cfg['ensembl75'] = pyensembl.EnsemblRelease(species=pyensembl.species.Species.register(
latin_name='homo_sapiens',
synonyms=['homo_sapiens'],
reference_assemblies={
    'GRCh37': (75, 75),
}),release=75)
cfg['ensembl'] = pyensembl.EnsemblRelease(species=pyensembl.species.Species.register(
latin_name='homo_sapiens',
synonyms=['homo_sapiens'],
reference_assemblies={
    'GRCh38': (95, 95),
}),release=95)
## gene annotations
if exists(cfg['dgene_annotp']):
    ## get annotations
    from rohan.dandage.io_dfs import get_colsubset2stats
    cfg['dgene_annot']=read_table(cfg['dgene_annotp'])
    cfg['dgene_annot_stats']=read_table(cfg['dgene_annot_statsp'])
    cfg['dgene_annot_stats']=set_index(cfg['dgene_annot_stats'],'gene subset')
    # sample sizes in the legends
    cfg['colgene_subset2classes']=cfg['dgene_annot_stats'].apply(lambda x: x.index,axis=0)[cfg['dgene_annot_stats'].apply(lambda x: ~pd.isnull(x),axis=0)].apply(lambda x: dropna(x),axis=0).to_dict()
    cfg['colgene_subset2classns']=cfg['dgene_annot_stats'][cfg['dgene_annot_stats'].apply(lambda x: ~pd.isnull(x),axis=0)].apply(lambda x: dropna(x),axis=0).to_dict()
    cfg['colgene_subset2classns']={k:[int(i) for i in cfg['colgene_subset2classns'][k]] for k in cfg['colgene_subset2classns']}

    cfg['colgene_subsetpre2colxys']={'paralog or singleton': ['paralog', 'singleton', 'unclassified'],
                              'heteromer or not':['heteromer','not heteromer'],
                              'homomer or not':['homomer','not homomer'],
                              'heteromer or homomer':['heteromer and homomer','homomer'],
                             }
    if exists(cfg['dparalog_annotp']):
        cfg['dparalog_annot']=read_table(cfg['dparalog_annotp'])
        cfg['dparalog_annot_stats'],cfg['colparalog_subset2classes'],cfg['colparalog_subset2classns']=get_colsubset2stats(cfg['dparalog_annot'],
                           [c for c in cfg['dparalog_annot'] if c.startswith('heteromer ') or c.startswith('homomer ')])
    else:
        print('warning: dparalog_annotp does not exists')
    ##vars
    cfg['cols_gene_subset']=list(cfg['colgene_subset2classes'].keys())

    cfg['gene_subset2colors']={
    'paralog':'#FF2A2A',
    'singleton':'#2A7FFF',
    'unclassified':'#C0C0C0',
    'heteromer':'#FF6600',
    'not heteromer':'#37C837',
    'homomer':'#00FF00',
    'not homomer':'#D4AA00',
    'heteromer and homomer':'#D45500',
    'essential':'#222222',
    'non-essential':'#111111',
    }
    cfg['colgene_subset2classcolors']={k:[cfg['gene_subset2colors'][s] for s in cfg['colgene_subset2classes'][k]] for k in cfg['colgene_subset2classes']}
    cfg['colparalog_subset2classcolors']={k:[cfg['gene_subset2colors'][s] for s in cfg['colparalog_subset2classes'][k]] for k in cfg['colparalog_subset2classes']}
## labels for merging
cfg['dn2label']={  'CS1','Wang et al.',
          'CS2','DepMap: CERES',
         'CS2.1','DepMap: unique alignments',
         'CS3','Shifrut et al.',}
## ints
cfg['ppitypes']=['all PPI','direct PPI']
## labels for subsets of data
cfg['dataset_cs']=['CS1','CS2','CS2.1','CS3']
cfg['dataset2cols']=ordereddict({
'CS': [
        'CS mean CS1',
        'CS mean CS2',
        'CS mean CS2.1',
        'CS mean CS3',
        ],
'FPKM (log2 scale)':['FPKM (log2 scale) mean'],
      })
cfg['dataset_cols']=[]
for k in cfg['dataset2cols']:
    cfg['dataset_cols']+=cfg['dataset2cols'][k]
cfg['colcellline']='cell line'
cfg['colgenename']='gene name'
cfg['cols_paralog_stats']=['evolutionary distance','% identity','dN/dS', 'dN', 'dS']
## plotting colors
cfg['dataset_cs2color']=dict(zip(cfg['dataset_cs'],['#FF00FF','#A9A9A9','#00FF00','#0000FF','#FF0000']))
boxprops = {'edgecolor': 'k', 'linewidth': 2, 'facecolor': 'w'}
lineprops = {'color': 'k', 'linewidth': 2}
cfg['boxplot_kwargs'] = dict({'boxprops': boxprops, 'medianprops': lineprops,
                       'whiskerprops': lineprops, 'capprops': lineprops,},)
# for jupyter notebook
for k in cfg:
    globals()[k] = cfg[k]