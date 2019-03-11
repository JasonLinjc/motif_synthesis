require(ggplot2)
require(ggseqlogo)
data(ggseqlogo_sample)
load("~/Documents/year2semB/motif_synthesis/JiecongData/motifDatabase.RData")
load("~/Documents/year2semB/motif_synthesis/JiecongData/HomodimerMotifDatabase.RData")
load("~/Documents/year2semB/motif_synthesis/JiecongData/DimerMotifDatabase.RData")
# ggseqlogo(pfms_dna$MA0031.1, method = 'prob')
# pfm2 = HomodimerMotifDatabase$Alx1
# rownames(pfm2) = c('A','C','G','T')
# ggseqlogo(pfm2, method = 'prob')
# length(DimerMotifFamily)
# n = keys(DimerMotifFamily)[1]
# DimerMotifFamily[[n]]

visualise_motifs <- function(fname, dname)
{
  dimer = list()
  j = 1
  for (i in 1:length(DimerMotifFamily))
  {
    motif_name = keys(DimerMotifFamily)[i]
    family_name = DimerMotifFamily[[motif_name]]
    if (family_name == fname)
    {
      dimer[[j]] <- motif_name
      j = j + 1
    }
  }
  vdimer = dimer[[5]]
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
fname = "Homeo_Tbox"
visualise_pred_dimer(paste("~/year2semB/motif_synthesis/pred_dimer/", fname,"/" ,sep=""))


# pfm = DimerMotifDatabase[[vdimer]]
# print(vdimer)
# rownames(pfm) = c('A','C','G','T')
# ggseqlogo(pfm, method = 'prob')

# a <- read.table("year2semB/motif_synthesis/pred_dimer/CEBPG_CREB3L1_pred.csv", header=FALSE, se[=','])
# data.matrix(a)
# rownames(a) = c('A','C','G','T')



