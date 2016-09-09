"""
Introduction to tensorflow
"""
import tensorflow as tf
#calling the default graph
graph = tf.get_default_graph()
#constant input values
x = tf.constant([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]],name='tensor1')
y = tf.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],name='tensor2')
#matrix multiplication
out = tf.matmul(x, y,name='output')
#finding operations in the default graph
for op in graph.get_operations():
    print(op.name)
#Const
#Const_1
#MatMul
#starting a session
sess=tf.Session()
#Initialzing variables
init=tf.initialize_all_variables()
sess.run(init)
output=sess.run(out)
print(output)
"""
[[ 2.  2.  2.]
 [ 2.  2.  2.]
 [ 2.  2.  2.]]
"""
"""summary writer will invoke tensorboard which is used to visualize the graph"""
summary_writer=tf.train.SummaryWriter('log_matmul_graph',sess.graph)