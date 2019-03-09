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

dimer = list()
j = 1
for (i in 1:length(DimerMotifFamily))
{
  motif_name = keys(DimerMotifFamily)[i]
  family_name = DimerMotifFamily[[motif_name]]
  if (family_name == "bHLH_Homeo")
  {
    dimer[[j]] <- motif_name
    j = j + 1
  }
}

vdimer = dimer[[3]]
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

# pfm = DimerMotifDatabase[[vdimer]]
# print(vdimer)
# rownames(pfm) = c('A','C','G','T')
# ggseqlogo(pfm, method = 'prob')

