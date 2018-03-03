from chatterbot import ChatBot
bot = ChatBot(
    'TARS',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    input_adapter='chatterbot.input.TerminalAdapter',
    output_adapter='chatterbot.output.TerminalAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        {
            'import_path': 'chatterbot.logic.BestMatch'
        },
        {
            'import_path': 'chatterbot.logic.LowConfidenceAdapter',
            'threshold': 0.65,
            'default_response': 'I am sorry, but I do not understand.'
        }
    ],
    trainer='chatterbot.trainers.ListTrainer',

    database='./database.sqlite3',
)
bot.train([
    'hello',
    'hi',
])
while True:
    try:
     print("---<Ask TARS a question>---")
     bot_input = bot.get_response(None)

    except(KeyboardInterrupt, EOFError, SystemExit):
        breaks
