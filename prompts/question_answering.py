import json


ALPACA_USER_PROMPT = """You are a helpful assistant. Please help me answer the query below. Keep your answer within 200 words.

# Query
{query}

Now, please provide your answer to this query.
# Answer
"""


URIAL_PROMPT = """# Instruction
Below is a list of conversations between a human and an AI assistant (you).
As an AI assistant, you will engage in conversations with users, responding to their queries which are presented under the heading "# Query".
Your responses should be entered under the heading "# Answer".
You excel in a wide range of tasks including, but not limited to, providing general information, conducting reasoning, engaging in role-play, creative writing, planning, and solving mathematical and coding problems.
Your responses should be well-structured, comprehensive, and aim to thoroughly address the user's query or problem at hand.
**Please keep your answer within 200 words.**

# Query
Can you tell me some common types of renewable energy sources?

# Answer
Certainly! Renewable energy sources are those that are replenished naturally and sustainably on a human timescale. Here are some common types of renewable energy sources:

1. Solar Energy: Energy harnessed from the sun. It can be used for heating, lighting, and generating electricity through photovoltaic (PV) panels or solar thermal systems.
2. Wind Energy: Energy generated from the natural movement of air. Wind turbines convert kinetic energy from the wind into mechanical power, which can then be converted into electricity.
3. Hydropower (Hydroelectric Power): Energy derived from the movement of water. This can be the flow of rivers, tidal movements, or wave power. Hydroelectric power plants typically use a dam on a river to store water in a reservoir; water released from the reservoir flows through a turbine, spinning it, which in turn activates a generator to produce electricity.
4. Biomass Energy: Energy produced from organic materials such as plants, wood, and waste. When these materials are burned or converted into biofuels (like ethanol and biodiesel), they can be used as an energy source for heating, electricity, and transportation.

Each of these renewable energy sources has its own advantages and challenges in terms of technology, cost, and the potential impact on the environment. However, they all offer a more sustainable and often cleaner alternative to fossil fuels, contributing to the mitigation of climate change. Please let me know if you have any other questions!

# Query
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

# Answer
Hi, I'm happy to help you. To solve the problem, let's think step by step:
1. In April, Natalia sold clips to 48 friends.
2. In May, she sold half as many as she did in April. So, the number of clips she sold in May = 48/2 = 24.
3. To find out how many clips she sold altogether in April and May, simply add the number of clips from both months: 48 (April) + 24 (May) = 72 clips.

Therefore, Natalia sold 72 clips altogether in April and May. Please feel free to ask me if you have any more questions!

# Query
What is the proper method for deep frying a turkey?

# Answer
Hello! I'm happy to help you with your question about deep frying a turkey! Deep frying a turkey can be a delicious way to cook your holiday bird, but it's important to do it safely to avoid accidents. Here's a step-by-step guide to deep frying a turkey:

1. Prepare the turkey: Thoroughly thaw and dry the turkey before frying. Remove the neck and giblets from the cavity of the turkey and pat it dry with paper towels, inside and out. This will help prevent oil splatter and ensure a crispy skin.
2. Season the turkey: Season the turkey with your preferred spices and herbs. You can use a dry rub or inject the seasoning directly into the meat.
3. Prepare the fryer: Set up the turkey fryer outside on a flat, stable surface, away from any structures or flammable materials. Fill the fryer with peanut or canola oil to the 1. recommended level, typically indicated on the fryer. Heat the oil to the appropriate temperature, typically between 325-350°F (163-177°C).
4. Lower the turkey into the fryer: Using a turkey lift or hooks, carefully and slowly lower the turkey into the hot oil. Make sure the turkey is fully submerged in the oil.
5. Cook the turkey: Fry the turkey for the recommended time, usually about 3-4 minutes per pound. Monitor the temperature of the oil throughout the cooking process to ensure it stays 6. within the recommended range.
6. Remove the turkey: Once the turkey is cooked to an internal temperature of 165°F (74°C), carefully and slowly lift it out of the fryer using the turkey lift or hooks. Place it on a wire rack or tray to drain any excess oil.

Remember to always prioritize safety when deep frying a turkey. Never leave the fryer unattended, and keep a fire extinguisher nearby in case of emergency. Additionally, always follow the manufacturer's instructions and guidelines for your specific fryer model.

# Query
{query}

# Answer
"""





ALPACA_PREFERENCE_SYSTEM_PROMPT = 'You are a helpful assistant tasked with evaluating the quality of different responses to a given query. For each query, you will receive two independent responses. Please tell me which response you think is better.'


ALPACA_PREFERENCE_USER_PROMPT = """Here is the given query.
# Query
{query}

Here are two independent responses (Response A and Response B) to the query.
# Responses
<Response A>{response1}</Response A>
<Response B>{response2}</Response B>

Now, please tell me which response you think is better. You only need to output A or B, with no other text. Please remember that your response should start with either A or B."""


##############################################################
# The following prompt is used when there is no system prompt.
# It was used to align with the model I trained
##############################################################
ALPACA_PREFERENCE_USER_PROMPT_ONLY = """You are a helpful assistant tasked with evaluating the quality of different responses to a given query. For each query, you will receive two independent responses. Please judge which response is better.

Here is the given query.
# Query
{query}

Here are two independent responses (Response A and Response B) to the query.
# Responses
<Response A>{response1}</Response A>
<Response B>{response2}</Response B>

Now, please judge which response is better. You only need to output A or B, with no other text.
Your output should be A if Response A is better or B if Response B is better.
Please remember that your response should start with either A or B
"""


###################
# few-shot instruct
###################

with open('few_shot_examples/alpaca_eval.json', 'r', encoding='utf-8') as f:
    alpaca_few_shot = json.load(f)


############
# base model
############
ALPACA_PREFERENCE_PROMPT_TWO_SHOT = """# Instruction
You are a helpful assistant tasked with evaluating the quality of different responses to a query. You will receive multiple queries, each accompanied by two independent responses. For each query, you should judge which response you think is better.
Queries are placed under "# Query". The two responses for each query are listed under "# Responses" as "Response A" and "Response B". Your outputs should be placed under "# Judgement".
You only need to output A or B, with no other text.

# Query
{query1}

# Responses
<Response A>{response11}</Response A>
<Response B>{response12}</Response B>

# Judgement
{judgement1}

# Query
{query2}

# Responses
<Response A>{response21}</Response A>
<Response B>{response22}</Response B>

# Judgement
{judgement2}

# Query
{{test_query}}

# Responses
<Response A>{{test_response1}}</Response A>
<Response B>{{test_response2}}</Response B>

# Judgement
"""

ALPACA_PREFERENCE_PROMPT_TWO_SHOT = ALPACA_PREFERENCE_PROMPT_TWO_SHOT.format(
    query1=alpaca_few_shot[0]["query"],
    response11=alpaca_few_shot[0]["accept"],
    response12=alpaca_few_shot[0]["reject"],
    judgement1="A",
    query2=alpaca_few_shot[2]["query"],
    response21=alpaca_few_shot[2]["reject"],
    response22=alpaca_few_shot[2]["accept"],
    judgement2="B",
)


################
# instruct model
################
ALPACA_PREFERENCE_USER_PROMPT_TWO_SHOT = """Here are some examples.
[example 1]
# Query
{query1}

# Responses
<Response A>{response11}</Response A>
<Response B>{response12}</Response B>

# Your output
{judgement1}

[example 2]
# Query
{query2}

# Responses
<Response A>{response21}</Response A>
<Response B>{response22}</Response B>

# Your output
{judgement2}


Below is the given query and two responses to it. Please tell me which response you think is better.
# Query
{{test_query}}

# Responses
<Response A>{{test_response1}}</Response A>
<Response B>{{test_response2}}</Response B>

You only need to output A or B, with no other text. Please remember that your output should start with either A or B.
# Your output
"""

ALPACA_PREFERENCE_USER_PROMPT_TWO_SHOT = ALPACA_PREFERENCE_USER_PROMPT_TWO_SHOT.format(
    query1=alpaca_few_shot[0]["query"],
    response11=alpaca_few_shot[0]["accept"],
    response12=alpaca_few_shot[0]["reject"],
    judgement1="A",
    query2=alpaca_few_shot[2]["query"],
    response21=alpaca_few_shot[2]["reject"],
    response22=alpaca_few_shot[2]["accept"],
    judgement2="B",
)





ALPACA_PREFERENCE_USER_PROMPT_ONLY_TWO_SHOT = """You are a helpful assistant tasked with evaluating the quality of different responses to a given query. For each query, you will receive two independent responses. Please judge which response is better.

Here are some examples.
[example 1]
# Query
{query1}

# Responses
<Response A>{response11}</Response A>
<Response B>{response12}</Response B>

# Your output
{judgement1}

[example 2]
# Query
{query2}

# Responses
<Response A>{response21}</Response A>
<Response B>{response22}</Response B>

# Your output
{judgement2}


Below is the given query and two responses to it. Please tell me which response you think is better.
# Query
{{test_query}}

# Responses
<Response A>{{test_response1}}</Response A>
<Response B>{{test_response2}}</Response B>

You only need to output A or B, with no other text. Please remember that your output should start with either A or B.
# Your output
"""

ALPACA_PREFERENCE_USER_PROMPT_ONLY_TWO_SHOT = ALPACA_PREFERENCE_USER_PROMPT_ONLY_TWO_SHOT.format(
    query1=alpaca_few_shot[0]["query"],
    response11=alpaca_few_shot[0]["accept"],
    response12=alpaca_few_shot[0]["reject"],
    judgement1="A",
    query2=alpaca_few_shot[2]["query"],
    response21=alpaca_few_shot[2]["reject"],
    response22=alpaca_few_shot[2]["accept"],
    judgement2="B",
)