library(dplyr)
library(phenoptr)
library(ggpubr)

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

polypList = c("Polyp40")

for (polypName in polypList){
  dataFolderName = paste("./data/",polypName,"/results/", sep='')
  areaFolderName = paste("./data",polypName,"/results/areas/", sep='')
  outputFolderNameForFiles = "./results/files/"
  outputFolderNameForFigures = "./results/figures/"  
  
  xCoordinatesColumns = c(9,11)
  yCoordinatesColumns = c(10,12)
  pixelAreaColumns = c(1)
  pixelDimensionColumns = c(9:14)
  phenotypes = c(22:37)
  epithelium = 38
  
  files <- list.files(dataFolderName, full.names=TRUE)
  data = read.table(file=files[1],header=TRUE,sep = '\t')
  data <- data[0,]
  
  
  pixelWidthMicron = 0.4984881
  pixelAreaMicron = pixelWidthMicron^2
  pixelAreaMm = pixelAreaMicron*1e-06
  if (exists("distances")) { rm(distances)}
  neighborhoodCounts = array(0, dim=c(length(phenotypes),length(phenotypes),2,5))
  searchRadius = 25
  for (f in files) {
    xOrigin = unlist(strsplit(unlist(strsplit(unlist(strsplit(f, split="['\\[']"))[2], split="[]]"))[1], split=","))[1]
    xOrigin = as.numeric(xOrigin)*pixelWidthMicron
    yOrigin = unlist(strsplit(unlist(strsplit(unlist(strsplit(f, split="['\\[']"))[2], split="[]]"))[1], split=","))[2]
    yOrigin = as.numeric(yOrigin)*pixelWidthMicron
    currentData = read.table(file=f,header=TRUE,sep = '\t')
    currentData[,pixelAreaColumns] = currentData[,pixelAreaColumns]*pixelAreaMicron
    currentData[,pixelDimensionColumns] = currentData[,pixelDimensionColumns]*pixelWidthMicron
    currentData[,xCoordinatesColumns] = currentData[,xCoordinatesColumns] + xOrigin
    currentData[,yCoordinatesColumns] = currentData[,yCoordinatesColumns] + yOrigin
    
    currentData$Tissue = "Other"
    currentData$Tissue[currentData[,epithelium]==0] = "Stroma"
    currentData$Tissue[currentData[,epithelium]==1] = "Epithelium"
    currentData$Phenotype = "Other"
    for (j in 1:length(phenotypes)){
      currentData$Phenotype[currentData[,phenotypes[j]]==1] = colnames(currentData)[phenotypes[j]]
    }
    data = rbind(data,currentData)
    
    csd <- currentData
    csd <- csd %>% filter(Phenotype!='other')
    csd$"Cell X Position" <- csd$X
    csd$"Cell Y Position" <- csd$Y
    currentDistances <- find_nearest_distance(csd, phenotypes = colnames(data)[phenotypes])
    if (exists("distances")) {distances = rbind(distances,currentDistances)}
    else {distances = currentDistances}

    for (i in 1:length(phenotypes)){
      for (j in 1:length(phenotypes)){
        if(i!=j){
          neighborhoodCounts[i,j,1,] = neighborhoodCounts[i,j,1,] + as.matrix(count_within(csd %>% filter(Tissue=="Epithelium"), from=colnames(data)[phenotypes][i], to=colnames(data)[phenotypes][j], radius=25))
          neighborhoodCounts[i,j,2,] = neighborhoodCounts[i,j,2,] + as.matrix(count_within(csd %>% filter(Tissue=="Stroma"), from=colnames(data)[phenotypes][i], to=colnames(data)[phenotypes][j], radius=25))
        }
      }
    }
  }
  neighborhoodCounts[,,,1] = searchRadius
  neighborhoodCounts[,,,2:5] = neighborhoodCounts[,,,2:5]/length(files)
  ############################
  outputNameAllNuclei = paste(outputFolderNameForFigures, "/", polypName, "_allNuclei.csv", sep='')
  dataForAlex = data[,c(1:14,40,41)]
  write.csv(dataForAlex, outputNameAllNuclei)
  ############################
  areaFiles <- list.files(areaFolderName, full.names=TRUE)
  area = read.table(file=areaFiles[1],header=TRUE,sep = '\t')
  area <- area[0,]
  for (f in areaFiles) {
    currentArea = read.table(file=f,header=TRUE,sep = '\t')
    currentArea = as.matrix(currentArea)
    currentArea = apply(currentArea,c(1,2),as.numeric)
    if ((currentArea[1,1]<1) | (currentArea[1,2]<1)) {
      cat(f)
      cat("\n")
      cat(currentArea)
      cat("\n")
    }
    area = rbind(area,currentArea)
  }
  
  if (exists("resultSheet1")) { rm(resultSheet1)}
  resultSheet1 = matrix(nrow = 1, ncol = 2)
  colnames(resultSheet1) = c("Id","Number of tiles")
  resultSheet1[1,1] = polypName
  resultSheet1[1,2] = length(files)
  outputName1 = paste(outputFolderNameForFiles, "/", polypName, "_1.csv", sep='')
  write.csv(resultSheet1, outputName1)

  if (exists("resultSheet2")) { rm(resultSheet2)}
  resultSheet2 = matrix(nrow = ncol(area)+1, ncol = length(phenotypes)+3)
  resultSheet2[1,1] = polypName
  resultSheet2[1,2] = "Stroma"
  resultSheet2[2,2] = "Epithelium"
  resultSheet2[3,2] = "Total"
  colnames(resultSheet2) = c("Id","Tissue",colnames(data)[phenotypes],"Total cells")
  for (j in 1:length(phenotypes)){
    resultSheet2[1,j+2] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==0))
    resultSheet2[2,j+2] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==1))
    resultSheet2[3,j+2] = sum(data[,phenotypes[j]]==1)
  }
  resultSheet2[1,ncol(resultSheet2)] = sum(data[,epithelium]==0)
  resultSheet2[2,ncol(resultSheet2)] = sum(data[,epithelium]==1)
  resultSheet2[3,ncol(resultSheet2)] = nrow(data)
  outputName2 = paste(outputFolderNameForFiles, "/", polypName, "_2.csv", sep='')
  write.csv(resultSheet2, outputName2)

  if (exists("resultSheet3")) { rm(resultSheet3)}
  resultSheet3 = matrix(nrow = ncol(area)+1, ncol = length(phenotypes)+2)
  resultSheet3[1,1] = polypName
  resultSheet3[1,2] = "Stroma"
  resultSheet3[2,2] = "Epithelium"
  resultSheet3[3,2] = "Total"
  colnames(resultSheet3) = c("Id","Tissue",colnames(data)[phenotypes])
  for (j in 1:length(phenotypes)){
    resultSheet3[1,j+2] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==0))/sum(data[,epithelium]==0)
    resultSheet3[2,j+2] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==1))/sum(data[,epithelium]==1)
    resultSheet3[3,j+2] = sum(data[,phenotypes[j]]==1)/nrow(data)
  }
  outputName3 = paste(outputFolderNameForFiles, "/", polypName, "_3.csv", sep='')
  write.csv(resultSheet3, outputName3)

  if (exists("resultSheet4")) { rm(resultSheet4)}
  resultSheet4 = matrix(nrow = ncol(area)+1, ncol = length(phenotypes)+4)
  resultSheet4[1,1] = polypName
  resultSheet4[1,2] = "Stroma"
  resultSheet4[2,2] = "Epithelium"
  resultSheet4[3,2] = "Total"
  colnames(resultSheet4) = c("Id","Tissue","Tissue area (mm^2)", colnames(data)[phenotypes],"Total cells")
  resultSheet4[1,3] = sum(area[,1])*pixelAreaMm
  resultSheet4[2,3] = sum(area[,2])*pixelAreaMm
  resultSheet4[3,3] = sum(area)*pixelAreaMm
  for (j in 1:length(phenotypes)){
    resultSheet4[1,j+3] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==0))/(sum(area[,1])*pixelAreaMm)
    resultSheet4[2,j+3] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==1))/(sum(area[,2])*pixelAreaMm)
    resultSheet4[3,j+3] = sum(data[,phenotypes[j]]==1)/(sum(area)*pixelAreaMm)
  }
  resultSheet4[1,ncol(resultSheet4)] = sum(data[,epithelium]==0)/(sum(area[,1])*pixelAreaMm)
  resultSheet4[2,ncol(resultSheet4)] = sum(data[,epithelium]==1)/(sum(area[,2])*pixelAreaMm)
  resultSheet4[3,ncol(resultSheet4)] = nrow(data)/(sum(area)*pixelAreaMm)
  outputName4 = paste(outputFolderNameForFiles, "/", polypName, "_4.csv", sep='')
  write.csv(resultSheet4, outputName4)

  # nb cells in given neighborhood
  if (exists("resultSheet5")) { rm(resultSheet5)}
  resultSheet5 = matrix(nrow = 2*dim(neighborhoodCounts)[1], ncol = dim(neighborhoodCounts)[1]+3)
  colnames(resultSheet5) = c("Id","Tissue","Phenotype",colnames(data)[phenotypes])
  resultSheet5[1,1] = polypName
  resultSheet5[1,2] = "Stroma"
  resultSheet5[1+length(phenotypes),2] = "Epithelium"
  for(i in 1:dim(neighborhoodCounts)[1]){
    resultSheet5[i,3] = colnames(data)[phenotypes][i]
    resultSheet5[i+length(phenotypes),3] = colnames(data)[phenotypes][i]
  }
  resultSheet5[1:dim(neighborhoodCounts)[1],(4:(dim(neighborhoodCounts)[1]+3))] = neighborhoodCounts[,,2,4]
  resultSheet5[(dim(neighborhoodCounts)[1]+1):nrow(resultSheet5),(4:(dim(neighborhoodCounts)[1]+3))] = neighborhoodCounts[,,1,4]
  outputName5 = paste(outputFolderNameForFiles, "/", polypName, "_5.csv", sep='')
  write.csv(resultSheet5, outputName5)

  # figures
  myColors <- gg_color_hue(length(phenotypes)+1)
  names(myColors) <- sort(c(unique(colnames(data)[phenotypes]),"Other"))
  
  # Counts 
  # Epithelium
  g1 <- ggplot(data %>% filter(Tissue=='Epithelium') %>% filter(Phenotype!='Other'), 
               aes(Phenotype))
  g1 <- g1 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
    theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
          axis.ticks.x=element_blank()) + 
    labs(title="Number of cells per phenotype in the epithelium") +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Tissue=='Epithelium') %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  # Stroma
  g2 <- ggplot(data %>% filter(Tissue=='Stroma') %>% filter(Phenotype!='Other'), 
               aes(Phenotype))
  g2 <- g2 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
    theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
          axis.ticks.x=element_blank()) + 
    labs(title="Number of cells per phenotype in the stroma")  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Tissue=='Stroma') %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  # all cells
  g3 <- ggplot(data %>% filter(Phenotype!='Other'), aes(Phenotype))
  g3 <- g3 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
    theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
          axis.ticks.x=element_blank()) + 
    labs(title="Total number of cells per phenotype") +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  
  figure <- ggarrange(g3, g1, g2,
                      common.legend = TRUE, legend = "right",
                      ncol = 1, nrow = 3)
  
  outputNameCounts = paste(outputFolderNameForFigures, polypName, "_1_NbCellsPerPhenotype.png", sep='')
  ggsave(filename = outputNameCounts,plot = figure, width = 8, height = 8, units = "in", device = "png")
  
  # Proportion of cells
  # Epithelium
  g1 <- ggplot(data %>% filter(Tissue=='Epithelium'), 
               aes(x = "", fill = factor(Phenotype)))
  g1 <- g1 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
    geom_bar(width = 1) +
    theme(axis.line = element_blank(), axis.text.x=element_blank(), panel.grid  = element_blank(),
          plot.title = element_text(hjust=0.5),axis.ticks=element_blank()) + 
    labs(fill="Phenotype", 
         x=NULL, 
         y=NULL, 
         title="Proportion of cells per phenotype in the epithelium")  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Tissue=='Epithelium'))$Phenotype)),names(myColors))])
  g1 <- g1 + coord_polar(theta = "y", start=0)
  # Stroma
  g2 <- ggplot(data %>% filter(Tissue=='Stroma'), 
               aes(x = "", fill = factor(Phenotype)))
  g2 <- g2 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
    geom_bar(width = 1) +
    theme(axis.line = element_blank(), axis.text.x=element_blank(), panel.grid  = element_blank(),
          plot.title = element_text(hjust=0.5),axis.ticks=element_blank()) + 
    labs(fill="Phenotype", 
         x=NULL, 
         y=NULL, 
         title="Proportion of cells per phenotype in the stroma")  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Tissue=='Stroma'))$Phenotype)),names(myColors))])
  g2 <- g2 + coord_polar(theta = "y", start=0)
  # All
  g3 <- ggplot(data, 
               aes(x = "", fill = factor(Phenotype)))
  g3 <- g3 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
    geom_bar(width = 1) +
    theme(axis.line = element_blank(), axis.text.x=element_blank(), panel.grid  = element_blank(),
          plot.title = element_text(hjust=0.5),axis.ticks=element_blank()) + 
    labs(fill="Phenotype", 
         x=NULL, 
         y=NULL, 
         title="Proportion of all cells per phenotype")  +
    scale_fill_manual(values=myColors[match(sort(unique(data$Phenotype)),names(myColors))])
  g3 <- g3 + coord_polar(theta = "y", start=0)
  
  figure <- ggarrange(g3, g1, g2,
                      common.legend = TRUE, legend = "right",
                      ncol = 1, nrow = 3)
  outputNameCounts = paste(outputFolderNameForFigures, polypName, "_2_ProportionOfCellsPerPhenotype.png", sep='')
  ggsave(filename = outputNameCounts,plot = figure, width = 8, height = 8, units = "in", device = "png")

  # distances
  data_with_distances <- bind_cols(data, distances)
  data_with_distances <- (data_with_distances %>% filter(Phenotype!="Other"))
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD4.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD4`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD4.Tbet.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD4.Tbet`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD4.RORgT.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD4.RORgT`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD4.FOXP3.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD4.FOXP3`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD4.RORgT.FOXP3.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD4.RORgT.FOXP3`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD8.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD8`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD8.Tbet.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD8.Tbet`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD8.RORgT.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD8.RORgT`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD8.RORgT.Tbet.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD8.RORgT.Tbet`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCD8.Tbet.FOXP3.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to CD8.Tbet.FOXP3`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToTbet.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to Tbet`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToRORgT.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to RORgT`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToFOXP3.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to FOXP3`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCytokeratin.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to Cytokeratin`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCytokeratin.RORgT.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to Cytokeratin.RORgT`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_distanceToCytokeratin.FOXP3.png", sep='')
  g <- ggplot(data_with_distances, aes(`Distance to Cytokeratin.FOXP3`, color=Phenotype)) +
    geom_density(size=1) + theme_minimal()  +
    scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
  ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")


  if (exists("cellDensities")) { rm(cellDensities)}
  cellDensities <- expand.grid(X=colnames(data)[phenotypes], Y=colnames(data)[phenotypes])
  cellDensities$Epithelium <- as.vector(neighborhoodCounts[,,1,4])
  cellDensities$Stroma <- as.vector(neighborhoodCounts[,,2,4])

  g1 <- ggplot(cellDensities, aes(Y, rev(X), fill= Stroma)) + geom_tile() +
    scale_fill_gradient(low="white", high="red", name="Number of cells") +
    scale_y_discrete(labels = rev(colnames(data)[phenotypes])) +
    theme(axis.title.x = element_blank(),axis.title.y = element_blank()) +
    ggtitle(label = "Average number of cells for each phenotype that have at least 1 neighbor of another phenotype within 25 um in the stroma")
  g2 <- ggplot(cellDensities, aes(Y, rev(X), fill= Epithelium)) + geom_tile() +
    scale_fill_gradient(low="white", high="red", name="Number of cells") +
    scale_y_discrete(labels = rev(colnames(data)[phenotypes])) +
    theme(axis.title.x = element_blank(),axis.title.y = element_blank()) +
    ggtitle(label = "Average number of cells for each phenotype that have at least 1 neighbor of another phenotype within 25 um in the epithelium")
  figure <- ggarrange(g1, g2,
                      common.legend = FALSE, legend = "right",
                      ncol = 1, nrow = 2)
  outputNameArea = paste(outputFolderNameForFigures, polypName, "_3_NbCellsInNeighborhood.png", sep='')
  ggsave(filename = outputNameArea,plot = figure, width = 18, height = 18, units = "in", device = "png")
  
}
