### Podcast Micro-Categories


# INITIALIZE --------------------------------------------------------------

setwd("C:/Users/jblauvelt/Desktop/Projects/podcast_micro_categories")

library(dplyr)
library(ggplot2)
library(stringr)
library(reshape2)
options(scipen = 30)

#*****
scen <- "2"
#*****

# LOAD MAP FILE -----------------------------------------------------------

# Load map file
map <- read.csv(paste0("raw/cat_maps/S", scen, " isubs.csv"), stringsAsFactors = F, na="")
names(map) <- c("usub_id", "usub_name", "closest_isub", "closest_isub2")

map$closest_isub <- str_replace(map$closest_isub, " \\([Ss]ports\\)$", "")
map$closest_isub2 <- str_replace(map$closest_isub2, " \\([Ss]ports\\)$", "")

map$closest_isub <- str_replace(map$closest_isub, "Self Help", "Self-Help")


# LOAD MODEL TOPIC PROBS --------------------------------------------------

probs <- read.csv(paste0("output/s", scen, "_doctopics.csv"), stringsAsFactors = F, na="")
probs$show_id <- 1:nrow(probs)



# LOAD LIST OF EPISODES AND THEIR ISUBS -----------------------------------

show_isubs <- read.csv(paste0("output/s", scen, "_shows_and_closest_topics.csv"),
                       stringsAsFactors = F, na="")
show_isubs <- show_isubs %>% filter(!is.na(usub_id))

stopifnot(nrow(probs) == nrow(show_isubs))
show_isubs$show_id <- 1:nrow(show_isubs)
isubs <- show_isubs[,c("show_id", "isub")]


# COMBINE -----------------------------------------------------------------

# Add isubs to model topic probabilities
probs <- cbind(probs, data_frame(isub=isubs[,c("isub")]))

# Melt probabilities to join format for joining
pr <- melt(probs, id.var=c("show_id", "isub"), variable.name="usub_id", value.name = "prob")
pr$usub_id <- as.integer(str_replace(pr$usub_id, "X", ""))

# Join in closest_isub to probs
      chk <- nrow(pr)
pr <- left_join(pr, map[,c("usub_id", "closest_isub")], by="usub_id")
      stopifnot(chk == nrow(pr))
      
# Sort by show and usub
pr <- pr %>% arrange(show_id, usub_id)

# Make sure all closest isubs are in the list of official isubs
stopifnot(length(setdiff(unique(pr$closest_isub), unique(pr$isub)))==0)
# sort(unique(show_isubs$isub))

# Filter to only where the isub matches the closest isub, to obtain
# the model's probability for the sub closest to its actual sub
pr <- pr %>% filter(!is.na(closest_isub) & isub == closest_isub)

# Some isubs map to two different usubs, so just take the usub
# with the higher probability
pr <- pr %>% arrange(show_id, -prob) %>% filter(!duplicated(show_id))


# JOIN PROBS BACK INTO SHOW LIST ------------------------------------------

# ll == "Log likelihood" because that's what we'll be calculating
ll <- left_join(show_isubs[,c("show_id", "podcast_name", "isub")],
                pr[,c("show_id", "prob")], by="show_id")

# For shows that didn't match to an isub (because no usubs matched to that isub),
# assign a smoothed probability of .0000024, which is the minimum probability
# seen in the table (min(ll$prob, na.rm=T)). We don't want to assign 0 because
# the log of 0 is -Inf
ll$prob[is.na(ll$prob)] <- .0000024


ggplot(ll, aes(prob)) + geom_histogram() + theme_bw()

# OUTPUT ------------------------------------------------------------------

# Log likelihood
print(paste0("Log-likelihood: ", round(sum(log(ll$prob, base=2)), 2)))

# Print number of usubs that map to isubs
print(paste0(sum(!is.na(map$closest_isub)), "/56 usubs were mapped to isubs"))
      
      
      
      