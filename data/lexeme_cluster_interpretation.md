# DuoBuddy: Semantic Cluster Definitions (Portuguese)

This document details the 81 Semantic Lexeme Clusters derived from Dataset B (2020 STAPLE Shared Task). These clusters function as the foundational latent dimensions for our User Capability Fingerprint, enabling a high-resolution mapping of learner proficiency. The granular cluster assignments for each lexeme are documented in:
`data/final/lexeme_cluster_results.csv`

---

## I. Core Behavior & Cognition

These clusters represent the most fundamental actions, intentions, and mental processes in language learning.

| Cluster ID | Theme                      | Key Examples                      |
| :--------- | :------------------------- | :-------------------------------- |
| **0**      | Finding & Discovery        | encontrar, achar                  |
| **2**      | Memory & Forgetting        | lembrar, esquecer, recordar       |
| **3**      | Existence & Occurrence     | haver, existir                    |
| **4**      | Emotions & Preferences     | amar, adorar, odiar, querido      |
| **5**      | Attempting Actions         | tentar, tentando                  |
| **6**      | Volition & Future Intent   | querer, planejar, ir, conseguir   |
| **13**     | Sensory Perception         | olhar, ouvir, ver, observar       |
| **14**     | Knowledge & Visiting       | saber, conhecer, visitar          |
| **15**     | Necessity & Requirement    | precisar, preciso, necessidade    |
| **47**     | Thinking & Decision Making | pensar, decidir, acreditar, supor |
| **52**     | Permission & Relinquishing | deixar, permitir                  |

---

## II. Communication & Social Interaction

This category is a key metric for the DuoBuddy matchmaking system to determine a user's "social learning potential."

| Cluster ID | Theme                         | Key Examples                             |
| :--------- | :---------------------------- | :--------------------------------------- |
| **8**      | Oral Communication            | falar, conversar, telefonar              |
| **9**      | Social Etiquette & Assistance | ajudar, desculpar, parabéns, convidar    |
| **10**     | Explaining & Presenting       | mostrar, responder, explicar, apresentar |
| **11**     | Narrating & Dictation         | dizer, contar, soletrar                  |
| **73**     | Psychological States & Effort | surpresa, esforço, sentimento, preocupar |
| **80**     | Messaging, Media & Stories    | mensagem, notícia, história, publicação  |

---

## III. Life Scenes & Physical Entities

Describes concrete objects and scenarios in daily life, representing a user's vocabulary breadth across different contexts.

| Cluster ID | Theme                  | Key Examples                           |
| :--------- | :--------------------- | :------------------------------------- |
| **1**      | Basic Needs (Dining)   | beber, comer, tomar, fumar             |
| **12**     | Colors & Visual Traits | rosa, vermelho, verde, azul            |
| **22**     | Body Parts & Health    | corpo, olhos, coração, perna, doença   |
| **33**     | Transportation         | carro, bicicleta, trem, avião, metrô   |
| **44**     | Finance & Payment      | pagar, gastar, dinheiro, caro, grátis  |
| **56**     | Geography & Nations    | brasil, américa, europa, região        |
| **58**     | Family & Peers         | família, filho, pai, amigo, criança    |
| **59**     | Weather & Nature       | chover, neve, sol, chuva, quente       |
| **61**     | Housing & Buildings    | casa, quarto, hotel, edifício          |
| **64**     | Animals & Flora        | gato, cachorro, cavalo, animal, flor   |
| **68**     | Appliances & Tools     | relógio, luz, cama, telefone, rádio    |
| **70**     | Home Interior          | cozinha, banheiro, janela, geladeira   |
| **71**     | Public Places          | restaurante, museu, biblioteca, igreja |
| **75**     | Food Ingredients       | maçã, sopa, chá, sal, cebola, doce     |
| **76**     | Clothing & Accessories | casaco, saia, chapéu, carteira         |

---

## IV. Grammar & Functional Structures

Reflects the user's mastery of the underlying structural "skeleton" of the language.

| Cluster ID | Theme                       | Key Examples                        |
| :--------- | :-------------------------- | :---------------------------------- |
| **16**     | State of Being (Estar)      | estar, estou, fica                  |
| **19**     | Possession (Ter)            | ter, tem, pertence                  |
| **32**     | Creation & Production       | fazer, cozinhar, produzir, criar    |
| **40**     | Indefinite Pronouns         | um, outro, qualquer, algum          |
| **41**     | Demonstrative Pronouns      | aquele, este, esse, nesta           |
| **42**     | Articles & Object Pronouns  | o, a, os, as, me, lhe               |
| **46**     | Abstract Pronouns & Fillers | isso, aquilo, tudo, algo, sim, nada |
| **50**     | Personal Pronouns           | eu, você, ele, nós, mim             |
| **51**     | Possessive Pronouns         | meu, seu, nosso, minha              |
| **54**     | Adverbs & Intensifiers      | não, também, exatamente, quase      |
| **55**     | Conjunctions                | com, ou, mas, pois, portanto        |

---

## V. Time, Space & Metrics

Used to locate the dimensions of when and where events occur.

| Cluster ID | Theme                 | Key Examples                         |
| :--------- | :-------------------- | :----------------------------------- |
| **17**     | Months                | janeiro, março, agosto, novembro     |
| **20**     | Seasons & Long Eras   | semana, século, inverno, verão, ano  |
| **21**     | Daily Time & Weekdays | ontem, quarta-feira, noite, cedo     |
| **25**     | Numbers & Ordinals    | primeiro, cinco, vinte, terceiro     |
| **35**     | Temporal Adverbs      | quando, logo, então, finalmente      |
| **43**     | Quantities & Units    | pouco, par, metade, hours, minute    |
| **49**     | Magnitude & Scale     | quanto, grande, pequeno, bastante    |
| **65**     | Spatial Position      | aonde, aqui, esquerda, frente, fundo |

---

## VI. Abstract Concepts & Professional Domains

Higher-level vocabulary groups typically mastered by advanced learners.

| Cluster ID | Theme                   | Key Examples                             |
| :--------- | :---------------------- | :--------------------------------------- |
| **37**     | Evaluation & Quality    | bom, melhor, perfeito, estranho, legal   |
| **38**     | Properties & Difficulty | difícil, fácil, rápido, perigoso         |
| **57**     | Science, Edu & Art      | ciência, professor, biologia, ensinar    |
| **60**     | Groups & Organizations  | população, equipe, clube, grupo          |
| **62**     | Occupations & Titles    | doutor, senhor, motorista, comandante    |
| **77**     | Logical Reasoning       | sobre, causa, resultado, impacto, useful |
| **78**     | Ambition & Planning     | objetivo, sonho, plano, carreira, ideia  |
| **79**     | Events & Activities     | viagem, exercício, reunião, jogo, exame  |

---

## Notes

- **<\*sf>**: Indicates "Stem Form." These have been mapped back to their root infinitives during feature calculation.
- **Miscellaneous Clusters**: Certain clusters (e.g., 23, 26, 28, 29, 30, 31, 34, 36, 39, 45, 48, 53, 63, 66, 67, 69, 72, 74) contain mixed motion verbs or general actions (ir, vir, levar, trazer) and are treated as a "General Activity" dimension in the fingerprint analysis.
