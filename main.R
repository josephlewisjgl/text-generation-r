library(gutenbergr)
library(poorman)
library(stringr)
library(tidytext)
library(purrr)
library(tokenizers)
library(comprehenr)
library(keras)
library(markovifyR)
library(knitr)
library(reticulate)
use_condaenv('r-reticulate')

# check an author exists in the Project Gutenberg database
gutenberg_works() %>% 
  filter(author == "Wells, H. G. (Herbert George)")

# select an author to query the database with 
selected_author <- 'Wells, H. G. (Herbert George)'

# query the database to return a dataframe with all works of the selected author
selected_author_books <- gutenberg_works(
  author == selected_author) %>% 
    gutenberg_download(meta_fields='title') %>% # add a title field from the metadata
  filter(text != "") %>% # remove empty rows 
  filter(!grepl("^ (I|V|X)( |\\.|[A-Z]{1}[a-z]*|of|the|and)*", text)) # remove chapter headings 

# replace special characters in the text field with spaces
df$text <- str_replace(df$text,"[^[:graph:]]", " ") 

# write to text file 
write.table(df$text, "selected_author_books.txt", sep="\t",row.names=FALSE)

# read in text file as a string for manipulation (for working offline)
fileName <- 'test.txt'
works_str <- readChar(fileName, file.info(fileName)$size) %>%
  gsub("\\\"", "", .) %>% # strip out backslash quotes
  gsub("\\n", " ", .) # strip out new line chars 

# tokenise into sentences and put into dataframe 
tokenised_sentences <- data.frame(tokenize_sentences(works_str))
colnames(tokenised_sentences) <- c('sent')

# force lower case 
tokenised_sentences$sent <- tolower(tokenised_sentences$sent)

# select only sentences below 240 characters 
df <- tokenised_sentences %>%
  filter(grep('.{0,240}', sent))

# build the markov model 
markov_model <-
  generate_markovify_model(
    input_text = df, 
    markov_state_size = 2L,
    max_overlap_total = 25,
    max_overlap_ratio = .5
  )

# generate the strings of tweet length
markovify_text(
  markov_model = markov_model,
  maximum_sentence_length = 240,
  output_column_name = selected_author,
  start_words = c('where'),
  count = 25,
  tries = 1000,
  only_distinct = TRUE,
  return_message = TRUE
)

# set up list of characters in the data 
fileName <- 'selected_author_books.txt'
works_str <- readChar(fileName, file.info(fileName)$size) %>%
  gsub("\\\"", "", .) %>% # strip out backslash quotes
  gsub("\\n", " ", .) %>% # strip out new line chars 
  iconv(., to='ASCII//TRANSLIT', sub='')%>%
  gsub('[^a-zA-Z0-9 -]', " ", .) %>%
  gsub('  ', " ", .) %>%
  tokenize_characters(lowercase = TRUE, strip_non_alphanum = TRUE, simplify = TRUE)

chars <- works_str %>%
  unique() %>%
  sort()
chars

max_length <- 30
dataset <- map(
  seq(1, length(works_str) - max_length - 1, by = 3), 
  ~list(sentence = works_str[.x:(.x + max_length - 1)], 
        next_char = works_str[.x + max_length])
)
dataset <- transpose(dataset)

vectorize <- function(data, chars, max_length){
  # array to store the current character in a sentence as a vector 
  x <- array(0, dim = c(length(data$sentence), max_length, length(chars)))
  
  # array to store the next character in a sentence as a vector 
  y <- array(0, dim = c(length(data$sentence), length(chars)))
  
  # set the right part of a vector to 1,0 depending on what character is present 
  for(i in 1:length(data$sentence)){
    x[i,,] <- sapply(chars, function(x){
      as.integer(x == data$sentence[[i]])
    })
    y[i,] <- as.integer(chars == data$next_char[[i]])
  }
  
  # return a list of the x,y arrays 
  list(y = y,
       x = x)
}

# perform the vectorisation
vectors <- vectorize(dataset, chars, max_length)

# create the keras model
create_model <- function(chars, max_length){
  # model is sequential 
  keras_model_sequential() %>%
    # add layers 
    layer_lstm(128, input_shape = c(max_length, length(chars))) %>%
    layer_dense(length(chars)) %>%
    layer_activation("softmax") %>% 
    compile(
      loss = "categorical_crossentropy", 
      optimizer = optimizer_rmsprop(lr = 0.01)
    )
}

# fitting the model 
fit_model <- function(model, vectors, epochs = 1){
  # fit the model to the vectors in batches for a set number of epochs (def 1)
  model %>% fit(
    vectors$x, vectors$y,
    batch_size = 128,
    epochs = epochs
  )
  NULL
}

# generate a tweet 
generate_tweet <- function(model, text, chars, max_length, diversity){
  
  # function to choose the next character for the phrase 
  choose_next_char <- function(preds, chars, div){
    # prediction is the log of predictions / the diversity value 
    preds <- log(preds) / div
    # convert the preds back from the log
    exp_preds <- exp(preds)
    preds <- exp_preds / sum(exp(preds))
    
    # get the index to insert the character into and return the character at
    # the predicted index 
    next_index <- rmultinom(1, 1, preds) %>% 
      as.integer() %>%
      which.max()
    chars[next_index]
  }
  
  # convert a sentence into an array so that the model can understand it 
  convert_sentence_to_data <- function(sentence, chars){
    # apply the function to transform a letter to it's vector representation
    x <- sapply(chars, function(x){
      as.integer(x == sentence)
    })
    array_reshape(x, c(1, dim(x)))
  }
  
  # use a sentence from the existing text to start with 
  start_index <- sample(1:(length(text) - max_length), size = 1)
  sentence <- text[start_index:(start_index + max_length - 1)]
  generated <- ""
  
  # while we still need characters for the phrase
  for(i in 1:(max_length * 3)){
    
    sentence_data <- convert_sentence_to_data(sentence, chars)
    
    # get the predictions for each next character
    preds <- predict(model, sentence_data)
    
    # choose the character
    next_char <- choose_next_char(preds, chars, diversity)
    
    # add it to the text and continue
    generated <- str_c(generated, next_char, collapse = "")
    sentence <- c(sentence[-1], next_char)
    
    if (sentence[-1] == " " & next_char == " ") {
      break
    }
  }
  
  generated
}

# running model iterations
iterate_model <- function(model, text, chars, max_length, 
                          diversity, vectors, iterations){
  
  # for each iteration
  for(iteration in 1:iterations){
    
    # print a message
    print(iteration)
    # fit the model
    fit_model(model, vectors)
    
    # print the diversity value 
    for(diversity in c(0.2, 0.5, 1)){
      
      message(sprintf("diversity: %f ---------------\n\n", diversity))
      
      # generate a tweet
      current_phrase <- 1:10 %>% 
        map_chr(function(x) generate_tweet(model,
                                            text,
                                            chars,
                                            max_length, 
                                            diversity))
      
      # print the current generated tweet
      message(current_phrase, sep="\n")
      message("\n\n")
      
    }
  }
  NULL
}

# create the model and begin iterating
model <- create_model(chars, max_length)
iterate_model(model, works_str, chars, max_length, diversity, vectors, 20)
## NULL

# build a df to hold results 
result <- data_frame(diversity = rep(c(0.2, 0.4, 0.6), 20)) %>%
  mutate(phrase = map_chr(diversity,
                          ~ generate_tweet(model, text, chars, max_length, .x))) %>%
  arrange(diversity)

# sample the results and show 
result %>%
  sample_n(10) %>%
  arrange(diversity) %>%
  kable()
