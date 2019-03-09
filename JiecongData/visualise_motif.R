require(ggplot2)
require(ggseqlogo)
data(ggseqlogo_sample)
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
print(dimer[[1]])

