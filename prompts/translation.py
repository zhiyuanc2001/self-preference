import json

with open('few_shot_examples/translation.json', 'r', encoding='utf-8') as f:
    translation_few_shot = json.load(f)




##################
# gen_translation
##################

TRANSLATION_USER_PROMPT = """You are an excellent translator, and you specialize in translating German into English. Given a piece of German text, please help translate it into English.
Here is the given German text.
# German
{german}

Now, please translate the German text into English. You only need to provide the English translation, with no other text.
# English
"""




TRANSLATION_PROMPT_TWO_SHOT = """# Instruction
You are an excellent translator, and you specialize in translating German into English. **Given a piece of German text, please translate it into English.**
The German texts are under "# German", and the corresponding English translations are under "# English".

# German
{german1}

# English
{english1}

# German
{german2}

# English
{english2}

# German
{{test_german}}

# English
"""

TRANSLATION_PROMPT_TWO_SHOT = TRANSLATION_PROMPT_TWO_SHOT.format(
    german1=translation_few_shot[0]['german'],
    english1=translation_few_shot[0]['accept'],
    german2=translation_few_shot[1]['german'],
    english2=translation_few_shot[1]['accept']
)




################
# gen_preference
################


TRANSLATION_PREFERENCE_USER_PROMPT = """You are a helpful assistant tasked with evaluating the quality of two different English translations of the same German text. For each German text, you will receive two independent English translations. Please judge which English translation is better.

Here is the German text.
# German
{german}

Here are two independent English translations (English A and English B) for the German text.
# English
<English A>{english1}</English A>
<English B>{english2}</English B>

Now, please judge which English translation is better. You only need to output A or B, with no other text. Please remember that your response should start with either A or B."""



TRANSLATION_PREFERENCE_PROMPT_TWO_SHOT = """# Instruction
You are a helpful assistant tasked with evaluating the quality of two different English translations of the same German text. For each German text, you will receive two independent English translations. Please judge which English translation is better.
The German texts are under "# German". The two independent English translations for each German text are under "# English", labeled as "English A" and "English B", respectively. Your outputs should be placed under "# Judgement".
You only need to output A or B, with no other text.

# German
{german1}

# English
<English A>{english11}</English A>
<English B>{english12}</English B>

# Judgement
{judgement1}

# German
{german2}

# English
<English A>{english21}</English A>
<English B>{english22}</English B>

# Judgement
{judgement2}

# German
{{test_german}}

# English
<English A>{{test_english1}}</English A>
<English B>{{test_english2}}</English B>

# Judgement
"""

TRANSLATION_PREFERENCE_PROMPT_TWO_SHOT = TRANSLATION_PREFERENCE_PROMPT_TWO_SHOT.format(
    german1=translation_few_shot[0]['german'],
    english11=translation_few_shot[0]['accept'],
    english12=translation_few_shot[0]['reject'],
    judgement1="A",
    german2=translation_few_shot[1]["german"],
    english21=translation_few_shot[1]['reject'],
    english22=translation_few_shot[1]["accept"],
    judgement2="B"
)

