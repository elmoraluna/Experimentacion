library("tidyverse")
library("fastDummies")

#Directorio
path_loc <- "D:/Google Drive/Universidad/Noveno/Seminario1/Pruebas"
setwd(path_loc)

#Lectura del dataset
df <-  read_csv2("student-mat.csv")

CambiarCategorica <- function(encontrado){
  if (identical(encontrado, "yes")){
    return(as.integer(1))
  }
  else if(identical(encontrado, "no")){
    return(as.integer(0))
  }
}

#Cambio de variables categóricas
table <- df %>% mutate(
  soporteColegio =  pmap_dbl(list(.$schoolsup), CambiarCategorica)) %>% mutate(
    soporteFamilia = pmap_dbl(list(.$famsup), CambiarCategorica)) %>% mutate(
      clasesExtra = pmap_dbl(list(.$paid), CambiarCategorica)) %>% mutate(
        actividades = pmap_dbl(list(.$activities), CambiarCategorica)) %>% mutate(
          guarderia = pmap_dbl(list(.$nursery), CambiarCategorica)) %>% mutate(
            superior = pmap_dbl(list(.$higher), CambiarCategorica)) %>% mutate(
              internetFijo = pmap_dbl(list(.$internet), CambiarCategorica)) %>% mutate(
                relacion = pmap_dbl(list(.$romantic), CambiarCategorica)) %>% select(
                  -c("schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"))

table <- table[, c(1:22, 26:33, 23, 24, 25)]

#One hot encoding
table <- dummy_cols(table, select_columns = c("school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian")) %>% 
  select(-c("school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"))

table <- table[, c(1:21, 25:50, 22, 23, 24)]

write_csv(table, "DatasetLimpio.csv")
