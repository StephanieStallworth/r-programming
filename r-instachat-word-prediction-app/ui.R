
# This is the user-interface definition of a Shiny web application. You can run the application by clicking 'Run App' above.
library(shiny)
library(shinythemes)
shinyUI(fluidPage(theme =shinytheme("superhero"),
        headerPanel("InstaChat Word Prediction"),
        sidebarPanel(
                h4("Input word or phrase and click 'Predict' to view next word suggestions"),
                textInput("text", label = "", 
                          value = "Stuck in my"),
                submitButton("PREDICT")
        ),
        
        mainPanel(
                h4("Twitter Prediction:"),
                verbatimTextOutput("twitter_next"),
                h4("Blogs Prediction:"),
                verbatimTextOutput("blogs_next"),
                h4("News Site Prediction:"),
                verbatimTextOutput("news_next"),
                h6("Next word predictions based on English text only and excludes numbers/special characters."),
                h6("Twitter, blog, and news data used to train algorithm can be downloaded at:"),
                h6(a("http://www.corpora.heliohost.org/"))
                
                )
        
        
                ))