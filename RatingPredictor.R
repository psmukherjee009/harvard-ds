# Title     : Rating predictor for Movies
# Objective : Predict Movie ratings
# Created by: p_s_m
# Created on: 5/13/2020

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
 # https://grouplens.org/datasets/movielens/10m/
 # http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                            title = as.character(title),
                                            genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Save edx to a file so that we can bypass the above steps.
str(edx)
dim(edx)
save(edx, file="ml-10M100K/edx.Rda")


########################## Visualization ##########################################
# Load the data
library(tidyverse)
library(caret)
library(data.table)
load("ml-10M100K/edx.Rda")

## The above code is provided to obtain the dataset
dim(edx)
str(edx)
edx %>%
     summarize(n_users = n_distinct(userId),
               n_movies = n_distinct(movieId))

## Distribution of ratings per movie
edx %>%
     dplyr::count(movieId) %>%
     ggplot(aes(n)) +
     geom_histogram(bins = 30, color = "black") +
     scale_x_log10() +
     ggtitle("Movies")
# We can conclude that movies not rated by atleast 25 users can be removed.

## Distribution of Ratings per user
edx %>%
     dplyr::count(userId) %>%
     ggplot(aes(n)) +
     geom_histogram(bins = 30, color = "black") +
     scale_x_log10() +
     ggtitle("Users")
# We can conclude that a user rating less than 25 movies can be removed

# Find most rated movies
edx %>% group_by(movieId, title) %>%
	summarize(count = n()) %>%
    arrange(desc(count))

# Find least rated movies
edx %>% group_by(movieId, title) %>%
	summarize(count = n()) %>%
    arrange(count)

###################### CleanUp #########################################
## Denormalize genre data
edx_denorm <- separate_rows(edx, genres, sep = "\\|")
head(edx_denorm)
tedx_dnorm <- table(edx_denorm$genres)
# 7 rows are listed with no genres "(no genres listed)"
names(tedx_dnorm)
edx_denorm %>% filter(genres == "(no genres listed)")
# Pull My Daisy (1958) is the only movie without a genre so we will remove it from our dataset
# We conclude that we can remove movie id 8606

###################
# Lets clean up the data with the conclusion we had above
# 1. Remove users rating less than 25 movies -- Removed 7551 users. 155124 movies removed. 2.29% decrease
# 2. Remove movies not rated by atleast 25 users + movie id 8606 -- Removed 30576 movies. 0.4 % decrease.
library(tidyverse)
library(caret)
library(data.table)
load("ml-10M100K/edx.Rda")
edx_clean <- edx
edx_v1 <- dim(edx_clean)[[1]]
str(edx_clean)
rm_users <- edx_clean %>% group_by(userId) %>% summarize(n=n()) %>% filter(n < 25) %>% pull(userId)
head(rm_users)
str(rm_users)
dim(edx_clean)
edx_clean <- edx_clean %>% filter(!(userId %in% rm_users))
dim(edx_clean)
edx_v2 <- dim(edx_clean)[[1]]
str(edx_clean)
edx_v2 - edx_v1
(edx_v2 - edx_v1)/edx_v1


rm_movieIds <- edx_clean %>% group_by(movieId) %>% filter(n() < 25) %>% pull(movieId)
rm_movieIds <- append(rm_movieIds, 8606)
str(rm_movieIds)
str(edx_clean)
edx_clean <- edx_clean %>%filter(!(movieId %in% rm_movieIds))
dim(edx_clean)
str(edx_clean)
head(edx_clean)

save(edx_clean, file="ml-10M100K/edx_clean.Rda")
############################ Matrix Factorization ##############################
library(tidyverse)
library(caret)
library(data.table)
load("ml-10M100K/edx_clean.Rda")

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx_clean$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx_clean[-test_index,]
test_set <- edx_clean[test_index,] %>%
			semi_join(train_set, by = "movieId") %>%
     		semi_join(train_set, by = "userId")

RMSE <- function(true_ratings, predicted_ratings) {
     sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Step 1: Average Movie Rating
mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse
rmse_results <- tibble(Method = "Step 1: Average Movie Rating", RMSE = naive_rmse)
rmse_results

## Step 2: Add Movie Effect on Rating
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))
movie_avgs

predicted_ratings <- mu_hat + test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effect_rmse <- RMSE(test_set$rating, predicted_ratings)
movie_effect_rmse
##

rmse_results <- rmse_results %>% add_row(Method = "Step 2: Add Movie Effect on Rating", RMSE = movie_effect_rmse)
rmse_results

## Step 3: Add User Effect on Rating
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)
user_effect_rmse <- RMSE(test_set$rating, predicted_ratings)
user_effect_rmse
rmse_results <- rmse_results %>% add_row(Method = "Step 2: Add User Effect on Rating", RMSE = user_effect_rmse)
rmse_results
save(rmse_results, file="ml-10M100K/rmse_results.Rda")


