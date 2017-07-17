import tensorflow as tf

with tf.device("/gpu:0"):
    a = tf.Variable(tf.ones(()))
    a = tf.square(a)
with tf.device("/gpu:1"):
    b = tf.Variable(tf.ones(()))
    b = tf.square(b)
with tf.device("/cpu:0"):
    loss = a+b
    
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
for i in range(10):
    loss0, _ = sess.run([loss, train_op])
    print("loss", loss0)
