import sys
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import preprocess, create_contexts_target, convert_one_hot
from my_ch03.simple_cbow import SimpleCBOW


window_size = 1
hidden_size = 3
batch_size = 3
max_epoch = 1000

text = "You say goodbye and I say hello."
corpus, w2i, i2w = preprocess(text)

vocab_size = len(w2i)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()
