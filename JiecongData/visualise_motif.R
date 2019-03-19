require(ggplot2)
require(ggseqlogo)
data(ggseqlogo_sample)
load("~/Documents/year2semB/motif_synthesis/JiecongData/motifDatabase.RData")
load("~/Documents/year2semB/motif_synthesis/JiecongData/HomodimerMotifDatabase.RData")
load("~/Documents/year2semB/motif_synthesis/JiecongData/DimerMotifDatabase.RData")
load("~/Documents/year2semB/motif_synthesis/JiecongData/ProcessedData/misc.RData")
# ggseqlogo(pfms_dna$MA0031.1, method = 'prob')
# pfm2 = HomodimerMotifDatabase$Alx1
# rownames(pfm2) = c('A','C','G','T')
# ggseqlogo(pfm2, method = 'prob')
# length(DimerMotifFamily)
# n = keys(DimerMotifFamily)[1]
# DimerMotifFamily[[n]]
get_rev_com <- function(motif_seq)
{
  return(matrix(rev(motif_seq),nrow=4))
}

visualise_motifs <- function(dname, isRC, case_type, olen)
{
  vdimer = dname
  print(vdimer)
  split_motif = strsplit(vdimer, "_")[[1]]
  motif1 = split_motif[1]
  motif2 = split_motif[2]
  pfm_motif1 = MotifDatabase[[motif1]]
  pfm_motif2 = MotifDatabase[[motif2]]
  if (is.null(pfm_motif1))
  {
    pfm_motif1 = HomodimerMotifDatabase[[motif1]]
  }
  if (is.null(pfm_motif2))
  {
    pfm_motif2 = HomodimerMotifDatabase[[motif2]]
  }
  pfm_dimer = DimerMotifDatabase[[vdimer]]
  
  if (isRC == 1)
  {
    pfm_dimer = get_rev_com(pfm_dimer)
  }
  # Case 1 : X-Y
  if (case_type == 2) {
    # Case 2 : Y-X
    t = pfm_motif2
    pfm_motif2 = pfm_motif1
    pfm_motif1 = t
  }else if (case_type == 3) {
    # Case 3 : y-X
    pfm_motif2 = get_rev_com(pfm_motif2)
    t = pfm_motif2
    pfm_motif2 = pfm_motif1
    pfm_motif1 = t
  }else {
    # Case 4 : X-y
    pfm_motif2 = get_rev_com(pfm_motif2)
  }
  rownames(pfm_motif1) = c('A','C','G','T')
  rownames(pfm_motif2) = c('A','C','G','T')
  rownames(pfm_dimer) = c('A','C','G','T')
  namelist = list('First Motif'=pfm_motif1, 'Second Motif'=pfm_motif2, 'Dimer'=pfm_dimer)
  aa = ggseqlogo(namelist, ncol = 1, method='prob')
  plot(aa)
}

# visualise_motifs("bZIP_bZIP", dname = "CEBPG_CREB3L1")


visualise_pred_dimer <- function(dimer_dir)
{
  print(dimer_dir)
  dimer_files = list.files(dimer_dir)
  # print(dimer_files)
  for (i in 1:length(dimer_files))
  {
    # print(dimer_files[i][1])
    pred_dimer = (dimer_files[i])
    a <- read.table(paste(dimer_dir, pred_dimer, sep=""), header=FALSE, sep=',')
    pfm_pred_dimer <- data.matrix(a)
    
    tempImagePath = paste(pred_dimer,'_TestingCase.png',sep='')
    png(file=tempImagePath, width=1800, height=2200, res=300)
    
    rownames(pfm_pred_dimer) = c('A','C','G','T')
    # print(filename)
    dimer = strsplit(pred_dimer, ".p")[[1]][1]
    motif1 = strsplit(dimer, "_")[[1]][1]
    motif2 = strsplit(dimer, "_")[[1]][2]
    pfm_motif1 = MotifDatabase[[motif1]]
    pfm_motif2 = MotifDatabase[[motif2]]
    if (is.null(pfm_motif1))
    {
      pfm_motif1 = HomodimerMotifDatabase[[motif1]]
    }
    if (is.null(pfm_motif2))
    {
      pfm_motif2 = HomodimerMotifDatabase[[motif2]]
    }
    pfm_dimer = DimerMotifDatabase[[dimer]]
    pfm_motif1 = MotifDatabase[[motif1]]
    pfm_motif2 = MotifDatabase[[motif2]]
    if (is.null(pfm_motif1))
    {
      pfm_motif1 = HomodimerMotifDatabase[[motif1]]
    }
    if (is.null(pfm_motif2))
    {
      pfm_motif2 = HomodimerMotifDatabase[[motif2]]
    }
    rownames(pfm_motif1) = c('A','C','G','T')
    rownames(pfm_motif2) = c('A','C','G','T')
    rownames(pfm_dimer) = c('A','C','G','T')
    namelist = list('First Motif'=pfm_motif1, 'Second Motif'=pfm_motif2, 'Predicted Dimer'=pfm_pred_dimer, 'True Dimer'=pfm_dimer)
    aa = ggseqlogo(namelist, ncol = 1, method='prob')
    plot(aa)
    dev.off()
  }
}

# fname = "Homeo_Tbox"
# visualise_pred_dimer(paste("~/year2semB/motif_synthesis/pred_dimer/", fname,"/" ,sep=""))

# pfm = DimerMotifDatabase[[vdimer]]s
# print(vdimer)
# rownames(pfm) = c('A','C','G','T')
# ggseqlogo(pfm, method = 'prob')

# a <- read.table("year2semB/motif_synthesis/pred_dimer/CEBPG_CREB3L1_pred.csv", header=FALSE, se[=','])
# data.matrix(a)
# rownames(a) = c('A','C','G','T')

for (i in 1:nrow(NewAlignmentExperimentDF))
{
  info = NewAlignmentExperimentDF[1,]
  dimer_name = info[['nameOut']]
  case_type = info[['case']]
  isRC_ = info[['isRC']]
  olen = info[['overlapLen']]
  print(info)
  visualise_motifs(dimer_name, isRC_, case_type, olen)
  break
}


