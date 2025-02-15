# Personality Prediction Using the Myers-Briggs (MBTI) and Keirsey-Model

Which type of a person are you? Given seven billion living people, it is a difficult question to answer. The leading personality type theory today classifies humanity into these 16 personality types on the basis of psychological types and 4 on the Keisey Model.

Machine Learning Algorithms can be used to learn from a user’s social
media data and infer his/her behavioral type.

The goal of this repository is to identify if there are behavioral patterns in the information
shared in social media that can be mapped with high precision into the psychological types of
the MBTI or the temperaments of Keirsey.

The proposed framework infers temperament types following the David Keirsey’s model, and psychological types based on the MBTI model. On the twitter data 4 models have been trained – Logistic Regression, SGD Classifier, Random Forest, K Nearest Neighbours with showed an accuracy of 77%, 78%, 81% and 79%.

# The 16 Personality Types by Myers-Briggs and Keirsey

![16-personality-types-infographic](https://user-images.githubusercontent.com/39180928/90257882-61143980-de65-11ea-9361-fd4c196273a1.jpg)

# Four Temperaments by Keirsey

David Keirsey expanded on the ancient study of temperament by Hippocrates and Plato. In his works, Keirsey used the names suggested by Plato: Artisan (iconic), Guardian (pistic), Idealist (noetic), and Rational (dianoetic). Keirsey divided the four temperaments into two categories (roles), each with two types (role variants). The resulting 16 types correlate with the 16 personality types described by Briggs and Myers (MBTI).

## Artisans
    Composer (ISFP)
    Crafter (ISTP)
    Performer (ESFP)
    Promoter (ESTP)

## Guardians
    Inspector (ISTJ)
    Protector (ISFJ)
    Provider (ESFJ)
    Supervisor (ESTJ)

## Idealists
    Champion (ENFP)
    Counselor (INFJ)
    Healer (INFP)
    Teacher (ENFJ)

## Rationals
    Architect (INTP)
    Fieldmarshal (ENTJ)
    Inventor (ENTP)
    Mastermind (INTJ)

# Myers–Briggs Type Indicator

The Myers–Briggs Type Indicator (MBTI) is an introspective self-report questionnaire indicating differing psychological preferences in how people perceive the world and make decisions.The test attempts to assign four categories: introversion or extraversion, sensing or intuition, thinking or feeling, judging or perceiving. One letter from each category is taken to produce a four-letter test result, like "ISTJ" or "ENFP". 

## Extraverting
    Initiating
    Expressive
    Gregarious
    Active
    Enthusiastic
    
## Introverting
    Receiving
    Contained
    Intimate
    Reflective
    Quiet

## Sensing
    Concrete
    Realistic
    Practical
    Experiential
    Traditional
    
## Intuiting
    Abstract
    Imaginative
    Conceptual
    Theoretical
    Original

## Thinking
    Logical
    Reasonable
    Questioning
    Critical
    Tough

## Feeling
    Empathetic
    Compassionate
    Accommodating
    Accepting
    Tender

## Judging
    Systematic
    Planful
    Early Starting
    Scheduled
    Methodical

## Perceiving
    Casual
    Open-ended
    Prompted
    Spontaneous
    Emergent

### In other words one can say that Keirsey Model plays an integrated role in MYer-Briggs but not solely into it.

# Some Plotting to portay the personality details from the mbti1.csv dataset

## Persons having such personalities

![dummy](https://user-images.githubusercontent.com/39180928/90258640-76d62e80-de66-11ea-8dc0-ff19c9d3ba7f.png)

## Error Rate versus the K-Vaule of K-Nearest Neighbour

![Capture](https://user-images.githubusercontent.com/39180928/90329195-2246c680-dfc0-11ea-9a1a-919154d1a995.PNG)

## Scatter Plot

![scatter](https://user-images.githubusercontent.com/39180928/90526588-350ef600-e18e-11ea-9a52-ba74144d9b4a.png)

## Strip Plot

![stripplot](https://user-images.githubusercontent.com/39180928/90664947-fc901a80-e268-11ea-9c25-0550a53fef37.png)

## Violin Plot

![vilon](https://user-images.githubusercontent.com/39180928/90804609-19dfea00-e338-11ea-91b5-8635297af00a.png)
