library(dplyr)
library(phenoptr)
library(ggpubr)

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

summarize_data <- function(){
    parameters = read.table(file='./parameters.txt',header=FALSE,sep = '\t')
    dataFolderName = parameters[1,2]
    outputFolderNameForFiles = parameters[2,2]
    outputFolderNameForFigures = parameters[3,2]
    dataName = parameters[4,2]
    pixelWidthMicron = as.numeric(parameters[5,2])
    xCoordinatesColumns = as.numeric(parameters[6,2])
    yCoordinatesColumns = as.numeric(parameters[7,2])
    epithelium = as.numeric(parameters[8,2])
    epithelium_id = as.numeric(parameters[9,2])
    stroma_id = as.numeric(parameters[10,2])
    phenotype_first_column = as.numeric(parameters[11,2])
    phenotype_last_column = as.numeric(parameters[12,2])
    phenotypes = c(phenotype_first_column:phenotype_last_column)
    searchRadius = as.numeric(parameters[12,2])

    if (grepl("\\", dataFolderName, fixed = TRUE) == TRUE){    
        dataFolderName = gsub("\\\\", "/", dataFolderName)
    }
    if (grepl("\\", outputFolderNameForFiles, fixed = TRUE) == TRUE){
        outputFolderNameForFiles = gsub("\\\\", "/", outputFolderNameForFiles)
    }
    if (grepl("\\", outputFolderNameForFigures, fixed = TRUE) == TRUE){
        outputFolderNameForFigures = gsub("\\\\", "/", outputFolderNameForFigures)
    }

    files <- list.files(dataFolderName, full.names=TRUE)
    for (f in files) {
        if (grepl("area", f, fixed = TRUE) == FALSE) {
            data = read.table(file=f,header=TRUE,sep = '\t')
            data <- data[0,]
            break
        }
    }
    pixelAreaMicron = pixelWidthMicron^2
    pixelAreaMm = pixelAreaMicron*1e-06
    for (f in files) {
        if (grepl("area", f, fixed = TRUE) == FALSE) {
            xOrigin = unlist(strsplit(unlist(strsplit(unlist(strsplit(f, split="['\\[']"))[2], split="[]]"))[1], split=","))[1]
            xOrigin = as.numeric(xOrigin)*pixelWidthMicron
            yOrigin = unlist(strsplit(unlist(strsplit(unlist(strsplit(f, split="['\\[']"))[2], split="[]]"))[1], split=","))[2]
            yOrigin = as.numeric(yOrigin)*pixelWidthMicron
            currentData = read.table(file=f,header=TRUE,sep = '\t')
            currentData[,xCoordinatesColumns] = currentData[,xCoordinatesColumns] + xOrigin
            currentData[,yCoordinatesColumns] = currentData[,yCoordinatesColumns] + yOrigin
    
            currentData$Tissue = "Other"
            currentData$Tissue[currentData[,epithelium]!=epithelium_id] = "Stroma"
            currentData$Tissue[currentData[,epithelium]==epithelium_id] = "Epithelium"
            currentData$Phenotype = "Other"
            for (j in 1:length(phenotypes)){
                currentData$Phenotype[currentData[,phenotypes[j]]==1] = colnames(currentData)[phenotypes[j]]
            }
            data = rbind(data,currentData)
    
            csd <- currentData
            csd <- csd %>% filter(Phenotype!='other')
            csd$"Cell X Position" <- csd$X
            csd$"Cell Y Position" <- csd$Y
        }
    }

    for (f in files) {
        if (grepl("area", f, fixed = TRUE) == TRUE) {
            area = read.table(file=f,header=TRUE,sep = '\t')
            area <- area[0,]
            break
        }
    }
    for (f in files) {
        if (grepl("area", f, fixed = TRUE) == TRUE) {
            currentArea = read.table(file=f,header=TRUE,sep = '\t')
            newArea = matrix(nrow = 1, ncol = 2)
            newArea[1] = currentArea[stroma_id]
            newArea[2] = currentArea[epithelium_id]
            area = rbind(area,newArea)
        }
    }
  
    resultSheet1 = matrix(nrow = 1, ncol = 2)
    colnames(resultSheet1) = c("Id","Number of tiles")
    resultSheet1[1,1] = dataName
    resultSheet1[1,2] = length(files) - sum(grepl("area", files, fixed = TRUE))
    outputName1 = paste(outputFolderNameForFiles, "/", dataName, "_nb_tiles.csv", sep='')
    write.csv(resultSheet1, outputName1, row.names=FALSE)

    resultSheet2 = matrix(nrow = 3, ncol = length(phenotypes)+3)
    resultSheet2[1,1] = dataName
    resultSheet2[1,2] = "Stroma"
    resultSheet2[2,2] = "Epithelium"
    resultSheet2[3,2] = "Total"
    colnames(resultSheet2) = c("Id","Tissue",colnames(data)[phenotypes],"Total cells")
    for (j in 1:length(phenotypes)){
        resultSheet2[1,j+2] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]!=epithelium_id))
        resultSheet2[2,j+2] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==epithelium_id))
        resultSheet2[3,j+2] = sum(data[,phenotypes[j]]==1)
    }
    resultSheet2[1,ncol(resultSheet2)] = sum(data[,epithelium]!=epithelium_id)
    resultSheet2[2,ncol(resultSheet2)] = sum(data[,epithelium]==epithelium_id)
    resultSheet2[3,ncol(resultSheet2)] = nrow(data)
    resultSheet2[which(is.na(resultSheet2))] = ''
    outputName2 = paste(outputFolderNameForFiles, "/", dataName, "_nb_cells.csv", sep='')
    write.csv(resultSheet2, outputName2, row.names=FALSE)

    resultSheet3 = matrix(nrow = 3, ncol = length(phenotypes)+2)
    resultSheet3[1,1] = dataName
    resultSheet3[1,2] = "Stroma"
    resultSheet3[2,2] = "Epithelium"
    resultSheet3[3,2] = "Total"
    colnames(resultSheet3) = c("Id","Tissue",colnames(data)[phenotypes])
    for (j in 1:length(phenotypes)){
        resultSheet3[1,j+2] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]!=epithelium_id))/sum(data[,epithelium]!=epithelium_id)
        resultSheet3[2,j+2] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==epithelium_id))/sum(data[,epithelium]==epithelium_id)
        resultSheet3[3,j+2] = sum(data[,phenotypes[j]]==1)/nrow(data)
    }
    resultSheet3[which(is.na(resultSheet3))] = ''
    outputName3 = paste(outputFolderNameForFiles, "/", dataName, "_proportion_cells.csv", sep='')
    write.csv(resultSheet3, outputName3, row.names=FALSE)

    if (sum(grepl("area", files, fixed = TRUE)) > 0){
        resultSheet4 = matrix(nrow = 3, ncol = length(phenotypes)+4)
        resultSheet4[1,1] = dataName
        resultSheet4[1,2] = "Stroma"
        resultSheet4[2,2] = "Epithelium"
        resultSheet4[3,2] = "Total"
        colnames(resultSheet4) = c("Id","Tissue","Tissue area (mm^2)", colnames(data)[phenotypes],"Total cells")
        resultSheet4[1,3] = sum(area[,1])*pixelAreaMm
        resultSheet4[2,3] = sum(area[,2])*pixelAreaMm
        resultSheet4[3,3] = sum(area)*pixelAreaMm
        for (j in 1:length(phenotypes)){
            resultSheet4[1,j+3] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]!=epithelium_id))/(sum(area[,1])*pixelAreaMm)
            resultSheet4[2,j+3] = sum((data[,phenotypes[j]]==1)&(data[,epithelium]==epithelium_id))/(sum(area[,2])*pixelAreaMm)
            resultSheet4[3,j+3] = sum(data[,phenotypes[j]]==1)/(sum(area)*pixelAreaMm)
        }
        resultSheet4[1,ncol(resultSheet4)] = sum(data[,epithelium]!=epithelium_id)/(sum(area[,1])*pixelAreaMm)
        resultSheet4[2,ncol(resultSheet4)] = sum(data[,epithelium]==epithelium_id)/(sum(area[,2])*pixelAreaMm)
        resultSheet4[3,ncol(resultSheet4)] = nrow(data)/(sum(area)*pixelAreaMm)
        outputName4 = paste(outputFolderNameForFiles, "/", dataName, "_density_cells.csv", sep='')
        resultSheet4[which(is.na(resultSheet4))] = ''
        write.csv(resultSheet4, outputName4, row.names=FALSE)
    }

    # figures
    myColors <- gg_color_hue(length(phenotypes)+1)
    names(myColors) <- sort(c(unique(colnames(data)[phenotypes]),"Other"))
  
    # Counts 
    # Epithelium
    g1 <- ggplot(data %>% filter(Tissue=='Epithelium') %>% filter(Phenotype!='Other'),aes(Phenotype))
    g1 <- g1 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
        theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
        labs(title="Number of cells per phenotype in the epithelium") +
        scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Tissue=='Epithelium') %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
    # Stroma
    g2 <- ggplot(data %>% filter(Tissue=='Stroma') %>% filter(Phenotype!='Other'),aes(Phenotype))
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
  
    figure <- ggarrange(g3, g1, g2,common.legend = TRUE, legend = "right",ncol = 1, nrow = 3)
  
    outputNameCounts = paste(outputFolderNameForFigures, dataName, "_1_NbCellsPerPhenotype.png", sep='')
    ggsave(filename = outputNameCounts,plot = figure, width = 8, height = 8, units = "in", device = "png")
  
    # Proportion of cells
    # Epithelium
    g1 <- ggplot(data %>% filter(Tissue=='Epithelium'),aes(x = "", fill = factor(Phenotype)))
    g1 <- g1 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
        geom_bar(width = 1) +
        theme(axis.line = element_blank(), axis.text.x=element_blank(), panel.grid  = element_blank(),
        plot.title = element_text(hjust=0.5),axis.ticks=element_blank()) + 
        labs(fill="Phenotype", x=NULL, y=NULL, title="Proportion of cells per phenotype in the epithelium")  +
        scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Tissue=='Epithelium'))$Phenotype)),names(myColors))])
    g1 <- g1 + coord_polar(theta = "y", start=0)
    # Stroma
    g2 <- ggplot(data %>% filter(Tissue=='Stroma'), aes(x = "", fill = factor(Phenotype)))
    g2 <- g2 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
        geom_bar(width = 1) +
        theme(axis.line = element_blank(), axis.text.x=element_blank(), panel.grid  = element_blank(),
        plot.title = element_text(hjust=0.5),axis.ticks=element_blank()) + 
        labs(fill="Phenotype", x=NULL, y=NULL, title="Proportion of cells per phenotype in the stroma")  +
        scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Tissue=='Stroma'))$Phenotype)),names(myColors))])
    g2 <- g2 + coord_polar(theta = "y", start=0)
    # All
    g3 <- ggplot(data, aes(x = "", fill = factor(Phenotype)))
    g3 <- g3 + geom_bar(aes(fill=Phenotype), width = 0.5) + 
        geom_bar(width = 1) +
        theme(axis.line = element_blank(), axis.text.x=element_blank(), panel.grid  = element_blank(),
        plot.title = element_text(hjust=0.5),axis.ticks=element_blank()) + 
        labs(fill="Phenotype", x=NULL, y=NULL, title="Proportion of all cells per phenotype")  +
        scale_fill_manual(values=myColors[match(sort(unique(data$Phenotype)),names(myColors))])
    g3 <- g3 + coord_polar(theta = "y", start=0)
  
    figure <- ggarrange(g3, g1, g2, common.legend = TRUE, legend = "right", ncol = 1, nrow = 3)
    outputNameCounts = paste(outputFolderNameForFigures, dataName, "_2_ProportionOfCellsPerPhenotype.png", sep='')
    ggsave(filename = outputNameCounts,plot = figure, width = 8, height = 8, units = "in", device = "png")
}




spatial_distribution_analysis <- function(){
    parameters = read.table(file='./parameters.txt',header=FALSE,sep = '\t')
    dataFolderName = parameters[1,2]
    outputFolderNameForFiles = parameters[2,2]
    outputFolderNameForFigures = parameters[3,2]
    dataName = parameters[4,2]
    pixelWidthMicron = as.numeric(parameters[5,2])
    xCoordinatesColumns = as.numeric(parameters[6,2])
    yCoordinatesColumns = as.numeric(parameters[7,2])
    epithelium = as.numeric(parameters[8,2])
    epithelium_id = as.numeric(parameters[9,2])
    stroma_id = as.numeric(parameters[10,2])
    phenotype_first_column = as.numeric(parameters[11,2])
    phenotype_last_column = as.numeric(parameters[12,2])
    phenotypes = c(phenotype_first_column:phenotype_last_column)
    searchRadius = as.numeric(parameters[12,2])

    if (grepl("\\", dataFolderName, fixed = TRUE) == TRUE){    
        dataFolderName = gsub("\\\\", "/", dataFolderName)
    }
    if (grepl("\\", outputFolderNameForFiles, fixed = TRUE) == TRUE){
        outputFolderNameForFiles = gsub("\\\\", "/", outputFolderNameForFiles)
    }
    if (grepl("\\", outputFolderNameForFigures, fixed = TRUE) == TRUE){
        outputFolderNameForFigures = gsub("\\\\", "/", outputFolderNameForFigures)
    }

    files <- list.files(dataFolderName, full.names=TRUE)
    data = read.table(file=files[1],header=TRUE,sep = '\t')
    data <- data[0,]
    pixelAreaMicron = pixelWidthMicron^2
    pixelAreaMm = pixelAreaMicron*1e-06
    neighborhoodCounts = array(0, dim=c(length(phenotypes),length(phenotypes),2,5))
        
    current_file = 1
    for (f in files) {
        if (grepl("area", f, fixed = TRUE) == FALSE) {
            xOrigin = unlist(strsplit(unlist(strsplit(unlist(strsplit(f, split="['\\[']"))[2], split="[]]"))[1], split=","))[1]
            xOrigin = as.numeric(xOrigin)*pixelWidthMicron
            yOrigin = unlist(strsplit(unlist(strsplit(unlist(strsplit(f, split="['\\[']"))[2], split="[]]"))[1], split=","))[2]
            yOrigin = as.numeric(yOrigin)*pixelWidthMicron
            currentData = read.table(file=f,header=TRUE,sep = '\t')
            currentData[,xCoordinatesColumns] = currentData[,xCoordinatesColumns] + xOrigin
            currentData[,yCoordinatesColumns] = currentData[,yCoordinatesColumns] + yOrigin
    
            currentData$Tissue = "Other"
            currentData$Tissue[currentData[,epithelium]!=epithelium_id] = "Stroma"
            currentData$Tissue[currentData[,epithelium]==epithelium_id] = "Epithelium"
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
            distance_columns = grepl('Distance', colnames(currentDistances))
            currentDistances = currentDistances[, distance_columns]
            if (current_file == 1) {
                distances = currentDistances
            }
            else {
                distances = rbind(distances,currentDistances)
            }

            for (i in 1:length(phenotypes)){
                for (j in 1:length(phenotypes)){
                    if(i!=j){
                        neighborhoodCounts[i,j,1,] = neighborhoodCounts[i,j,1,] + as.matrix(count_within(csd %>% filter(Tissue=="Epithelium"), from=colnames(data)[phenotypes][i], to=colnames(data)[phenotypes][j], radius=searchRadius))
                        neighborhoodCounts[i,j,2,] = neighborhoodCounts[i,j,2,] + as.matrix(count_within(csd %>% filter(Tissue=="Stroma"), from=colnames(data)[phenotypes][i], to=colnames(data)[phenotypes][j], radius=searchRadius))
                    }
                }
            }
            current_file = current_file + 1
        }
    }
    neighborhoodCounts[,,,1] = searchRadius
    neighborhoodCounts[,,,2:5] = neighborhoodCounts[,,,2:5]/length(files)


    # nb cells in given neighborhood
    if (exists("resultSheet5")) { rm(resultSheet5)}
    resultSheet5 = matrix(nrow = 2*dim(neighborhoodCounts)[1], ncol = dim(neighborhoodCounts)[1]+3)
    colnames(resultSheet5) = c("Id","Tissue","Phenotype",colnames(data)[phenotypes])
    resultSheet5[1,1] = dataName
    resultSheet5[1,2] = "Stroma"
    resultSheet5[1+length(phenotypes),2] = "Epithelium"
    for(i in 1:dim(neighborhoodCounts)[1]){
        resultSheet5[i,3] = colnames(data)[phenotypes][i]
        resultSheet5[i+length(phenotypes),3] = colnames(data)[phenotypes][i]
    }
    resultSheet5[1:dim(neighborhoodCounts)[1],(4:(dim(neighborhoodCounts)[1]+3))] = neighborhoodCounts[,,2,4]
    resultSheet5[(dim(neighborhoodCounts)[1]+1):nrow(resultSheet5),(4:(dim(neighborhoodCounts)[1]+3))] = neighborhoodCounts[,,1,4]
    resultSheet5[which(is.na(resultSheet5))] = ''
    outputName5 = paste(outputFolderNameForFiles, "/", dataName, "_nb_Cells_neighborhood.csv", sep='')
    write.csv(resultSheet5, outputName5, row.names=FALSE)

    # figures
    myColors <- gg_color_hue(length(phenotypes)+1)
    names(myColors) <- sort(c(unique(colnames(data)[phenotypes]),"Other"))
  

    # distances
    data_with_distances <- bind_cols(data, distances)
    data_with_distances <- (data_with_distances %>% filter(Phenotype!="Other"))
    colnames(data_with_distances) = gsub(" ", ".", colnames(data_with_distances))
    colnames(distances) = gsub(" ", ".", colnames(distances))
    for(i in 1:ncol(distances)){
        outputNameArea = paste(outputFolderNameForFigures, dataName, "_", colnames(distances[i]), ".png", sep='')
        g <- ggplot(data_with_distances, aes_string(colnames(distances[i]), color="Phenotype")) +
            geom_density(size=1) + theme_minimal()  +
            scale_fill_manual(values=myColors[match(sort(unique((data %>% filter(Phenotype!='Other'))$Phenotype)),names(myColors))])
        ggsave(filename = outputNameArea,plot = g, width = 8, height = 8, units = "in", device = "png")
    }

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
    figure <- ggarrange(g1, g2, common.legend = FALSE, legend = "right", ncol = 1, nrow = 2)
    outputNameArea = paste(outputFolderNameForFigures, dataName, "_3_NbCellsInNeighborhood.png", sep='')
    ggsave(filename = outputNameArea,plot = figure, width = 18, height = 18, units = "in", device = "png")
}




