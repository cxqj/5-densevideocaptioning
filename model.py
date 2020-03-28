"""
Model definition
Implementation of dense captioning model in the paper "Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning" by Jingwen Wang et al. in CVPR, 2018.
The code looks complicated since we need to handle some "dynamic" part of the graph
"""

import tensorflow as tf

class CaptionModel(object):

    def __init__(self, options):
        self.options = options
        # 用于初始化词嵌入矩阵
        self.initializer = tf.random_uniform_initializer(
            minval = - self.options['init_scale'],  # 0.08
            maxval = self.options['init_scale'])

        tf.set_random_seed(options['random_seed'])

    """ 
    build video feature embedding
    """
    def build_video_feat_embedding(self, video_feat, reuse=False):
        with tf.variable_scope('video_feat_embed', reuse=reuse) as scope:
            video_feat_embed = tf.contrib.layers.fully_connected(
                inputs=video_feat,
                num_outputs=self.options['word_embed_size'],    # 512
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
        return video_feat_embed

    """
    build word embedding for each word in a caption
    """
    def build_caption_embedding(self, caption, reuse=False):
        with tf.variable_scope('word_embed', reuse=reuse):
            embed_map = tf.get_variable(
                name='map',
                shape=(self.options['vocab_size'], self.options['word_embed_size']),   # 4414 X 512
                initializer=self.initializer  # 初始化此嵌入矩阵
            )
            # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
            #tf.nn.embedding_lookup（params, ids）:params可以是张量也可以是数组等，id就是对应的索引，
            caption_embed = tf.nn.embedding_lookup(embed_map, caption)# 从词嵌入表中查询caption中单词对应的embedding信息
        return caption_embed

#-----------------------------------------------------------------------------------------------------------------------#    
    """
    Build graph for proposal generation (inference)
    """
    def build_proposal_inference(self, reuse=False):
        inputs = {}
        outputs = {}

        # this line of code is just a message to inform that batch size should be set to 1 only
        batch_size = 1

        #******************** Define Proposal Module ******************#

        ## dim1: batch, dim2: video sequence length, dim3: video feature dimension
        ## video feature sequence

        # forward
        video_feat_fw = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat_fw')
        inputs['video_feat_fw'] = video_feat_fw  # (B,T,C)

        # backward
        video_feat_bw = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat_bw')
        inputs['video_feat_bw'] = video_feat_bw  # (B,T,C)
        
        
        rnn_cell_video_fw = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )
        
        
        rnn_cell_video_bw = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )

        with tf.variable_scope('proposal_module', reuse=reuse) as proposal_scope:

            '''video feature sequence encoding: forward pass
            '''
            with tf.variable_scope('video_encoder_fw', reuse=reuse) as scope:
                sequence_length = tf.expand_dims(tf.shape(video_feat_fw)[1], axis=0)
                initial_state = rnn_cell_video_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
                
                rnn_outputs_fw, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell_video_fw, 
                    inputs=video_feat_fw, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
                
            # (T,512)   
            rnn_outputs_fw_reshape = tf.reshape(rnn_outputs_fw, [-1, self.options['rnn_size']], name='rnn_outputs_fw_reshape')

            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal_fw', reuse=reuse) as scope:
                logit_output_fw = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_fw_reshape, 
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )   # (T,120) 预测每一个anchor属于动作提议的概率

            '''video feature sequence encoding: backward pass
            '''
            with tf.variable_scope('video_encoder_bw', reuse=reuse) as scope:
                #sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
                sequence_length = tf.expand_dims(tf.shape(video_feat_bw)[1], axis=0)
                initial_state = rnn_cell_video_bw.zero_state(batch_size=batch_size, dtype=tf.float32)
                
                rnn_outputs_bw, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell_video_bw, 
                    inputs=video_feat_bw, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
                
                
            rnn_outputs_bw_reshape = tf.reshape(rnn_outputs_bw, [-1, self.options['rnn_size']], name='rnn_outputs_bw_reshape')

            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal_bw', reuse=reuse) as scope:
                logit_output_bw = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_bw_reshape,
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )  # (T,120)

        # score
        proposal_score_fw = tf.sigmoid(logit_output_fw, name='proposal_score_fw')
        proposal_score_bw = tf.sigmoid(logit_output_bw, name='proposal_score_bw')
        
        # outputs from proposal module
        outputs['proposal_score_fw'] = proposal_score_fw
        outputs['proposal_score_bw'] = proposal_score_bw
        outputs['rnn_outputs_fw'] = rnn_outputs_fw_reshape
        outputs['rnn_outputs_bw'] = rnn_outputs_bw_reshape


        return inputs, outputs

    """
    Build graph for caption generation (inference)
    Surprisingly, I found using beam search leads to worse meteor score on ActivityNet Captions dataset; similar observation has been found by other dense captioning papers
    I do not use beam search when generating captions
    """
    def build_caption_greedy_inference(self, reuse=False):
        inputs = {}
        outputs = {}

        # proposal feature sequences (the localized proposals/events can be of different length, I set a 'max_proposal_len' to make it easy for GPU processing)
        proposal_feats = tf.placeholder(tf.float32, [None, self.options['max_proposal_len'], self.options['video_feat_dim']])  # (N,110,500)
        # combination of forward and backward hidden state, which encode event context information
        event_hidden_feats = tf.placeholder(tf.float32, [None, 2*self.options['rnn_size']])  # (N,1024)

        inputs['event_hidden_feats'] = event_hidden_feats   # (N,1024)
        inputs['proposal_feats'] = proposal_feats  # (N,110,500)

        # batch size for inference, depends on how many proposals are generated for a video
        eval_batch_size = tf.shape(proposal_feats)[0]   # N
        
        # intialize the rnn cell for captioning
        rnn_cell_caption = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )

        def get_rnn_cell():
            return tf.contrib.rnn.LSTMCell(num_units=self.options['rnn_size'], state_is_tuple=True, initializer=tf.orthogonal_initializer())

        # multi-layer LSTM
        multi_rnn_cell_caption = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(self.options['num_rnn_layers'])], state_is_tuple=True)

        # start word
        # 置N个proposal的起始都为START对应的单词索引
        
        word_id = tf.fill([eval_batch_size], self.options['vocab']['<START>'])
        word_id = tf.to_int64(word_id)
        word_ids = tf.expand_dims(word_id, axis=-1)  # (N,1)

        # probability (confidence) for the predicted word
        # 初始化所有单词的置信度为1
        word_confidences = tf.expand_dims(tf.fill([eval_batch_size], 1.), axis=-1)

        # initial state of caption generation
        initial_state = multi_rnn_cell_caption.zero_state(batch_size=eval_batch_size, dtype=tf.float32)
        state = initial_state

        with tf.variable_scope('caption_module', reuse=reuse) as caption_scope:

            # initialize memory cell and hidden output, note that the returned state is a tuple containing all states for each cell in MultiRNNCell
            state = multi_rnn_cell_caption.zero_state(batch_size=eval_batch_size, dtype=tf.float32)

            # proposal_feats_reshape : (N,110,500)--->(Nx110,500)
            proposal_feats_reshape = tf.reshape(proposal_feats, [-1, self.options['video_feat_dim']], name='video_feat_reshape')


            ## the caption data should be prepared in equal length, namely, with length of 'caption_seq_len'
            ## use caption mask data to mask out loss from sequence after end of token (<END>)
            # only the first loop create variable, the other loops reuse them, need to give variable scope name to each variable, otherwise tensorflow will create a new one
            for i in range(self.options['caption_seq_len']-1):

                if i > 0:
                    caption_scope.reuse_variables()

                # word embedding
                word_embed = self.build_caption_embedding(word_id)

                # get attention, receive both hidden state information (previous generated words) and video feature
                # state[:, 1] return all hidden states for all cells in MultiRNNCell
                h_state = tf.concat([s[1] for s in state], axis=-1)  # (N,1024)
                h_state_tile = tf.tile(h_state, [1, self.options['max_proposal_len']])  # (N,110x1024)
                # (N,110x1024)-->(Nx110,1024)
                h_state_reshape = tf.reshape(h_state_tile, [-1, self.options['num_rnn_layers']*self.options['rnn_size']])
                
                # repeat to match each feature vector in the localized proposal
                # (N,110x1024)-->(Nx110,1024)
                event_hidden_feats_tile = tf.tile(event_hidden_feats, [1, self.options['max_proposal_len']])  
                event_hidden_feats_reshape = tf.reshape(event_hidden_feats_tile, [-1, 2*self.options['rnn_size']])

                # (Nx110,500) + (Nx110,1024) + (Nx110,1024) = (Nx110,2548)
                feat_state_concat = tf.concat([proposal_feats_reshape, h_state_reshape, event_hidden_feats_reshape], axis=-1, name='feat_state_concat')
                #feat_state_concat = tf.concat([tf.reshape(tf.tile(word_embed, [1, self.options['max_proposal_len']]), [-1, self.options['word_embed_size']]), proposal_feats_reshape, h_state_reshape, event_hidden_feats_reshape], axis=-1, name='feat_state_concat')



                # use a two-layer network to model temporal soft attention over proposal feature sequence when predicting next word (dynamic)
                with tf.variable_scope('attention', reuse=reuse) as attention_scope:
                    attention_layer1 = tf.contrib.layers.fully_connected(
                        inputs = feat_state_concat,
                        num_outputs = self.options['attention_hidden_size'],
                        activation_fn = tf.nn.tanh,
                        weights_initializer = tf.contrib.layers.xavier_initializer()
                    ) 
                    attention_layer2 = tf.contrib.layers.fully_connected(
                        inputs = attention_layer1,
                        num_outputs = 1,
                        activation_fn = None,
                        weights_initializer = tf.contrib.layers.xavier_initializer()
                    )  # (Nx110,2548)-->(Nx110,512)-->(Nx110,1)

                # reshape to match
                # (Nx110,1)-->(N,110)-->(N,1,110)
                attention_reshape = tf.reshape(attention_layer2, [-1, self.options['max_proposal_len']], name='attention_reshape')
                attention_score = tf.nn.softmax(attention_reshape, dim=-1, name='attention_score')
                attention = tf.reshape(attention_score, [-1, 1, self.options['max_proposal_len']], name='attention')

                # attended video feature
                # (N,1,110)x(N,110,500) = (N,1,500)-->(N,500)
                attended_proposal_feat = tf.matmul(attention, proposal_feats, name='attended_proposal_feat')
                attended_proposal_feat_reshape = tf.reshape(attended_proposal_feat, [-1, self.options['video_feat_dim']], name='attended_proposal_feat_reshape')

                # whether to use proposal contexts to help generate the corresponding caption
                if self.options['no_context']:
                    proposal_feats_full = attended_proposal_feat_reshape
                else:
                    # whether to use gating function to combine the proposal contexts
                    if self.options['context_gating']:
                        # model a gate to weight each element of context and feature
                        attended_proposal_feat_reshape = tf.nn.tanh(attended_proposal_feat_reshape)
                        with tf.variable_scope('context_gating', reuse=reuse):
                            '''
                            context_feats_transform = tf.contrib.layers.fully_connected(
                                inputs=event_hidden_feats,
                                num_outputs=self.options['video_feat_dim'],
                                activation_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer()
                            )
                            '''
                            context_feats_transform = event_hidden_feats  # (N,1024)

                            # (N,500)-->(N,1024)
                            proposal_feats_transform = tf.contrib.layers.fully_connected(
                                inputs = attended_proposal_feat_reshape,  
                                num_outputs = 2*self.options['rnn_size'],
                                activation_fn = tf.nn.tanh,
                                weights_initializer = tf.contrib.layers.xavier_initializer()
                            )
                            
                           
                            gate = tf.contrib.layers.fully_connected(
                                inputs=tf.concat([word_embed, h_state, context_feats_transform, proposal_feats_transform], axis=-1),
                                num_outputs=2*self.options['rnn_size'],
                                activation_fn=tf.nn.sigmoid,
                                weights_initializer=tf.contrib.layers.xavier_initializer()
                            )   # (N,1024)

                            # (N,1024)x(N,1024),相当于对每个通道加权
                            gated_context_feats = tf.multiply(context_feats_transform, gate)
                            gated_proposal_feats = tf.multiply(proposal_feats_transform, 1.-gate)
                            proposal_feats_full = tf.concat([gated_context_feats, gated_proposal_feats], axis=-1)  # (N,2048)
                            
                    else:
                        proposal_feats_full = tf.concat([event_hidden_feats, attended_proposal_feat_reshape], axis=-1)

                # proposal feature embedded into word space
                proposal_feat_embed = self.build_video_feat_embedding(proposal_feats_full)  # (N,2048)-->(N,512)

                # get next state
                # caption_output: (N,512)
                # state:
                #     0: (N,512)
                #     1: (N,512)
                caption_output, state = multi_rnn_cell_caption(tf.concat([proposal_feat_embed, word_embed], axis=-1), state)

                # predict next word
                with tf.variable_scope('logits', reuse=reuse) as logits_scope:
                    logits = tf.contrib.layers.fully_connected(
                        inputs=caption_output,
                        num_outputs=self.options['vocab_size'],
                        activation_fn=None
                    )  # (N,4414)

                softmax = tf.nn.softmax(logits, name='softmax') # (N,4414)
                word_id = tf.argmax(softmax, axis=-1)  # (N,) , 用argmax取id
                word_confidence = tf.reduce_max(softmax, axis=-1)  # (N,) ， 用reduce_max 取置信度
                
                # 逐渐保存生成的结果
                # word_ids : (N,1)--->(N,2)--->(N,3)........
                word_ids = tf.concat([word_ids, tf.expand_dims(word_id, axis=-1)], axis=-1)
                
                # word_confidences : (N,1)-->(N,2)-->(N,3).......
                word_confidences = tf.concat([word_confidences, tf.expand_dims(word_confidence, axis=-1)], axis=-1)

        #sentence_confidences = tf.reduce_sum(tf.log(tf.clip_by_value(word_confidences, 1e-20, 1.)), axis=-1)
        word_confidences = tf.log(tf.clip_by_value(word_confidences, 1e-20, 1.))

        outputs['word_ids'] = word_ids
        outputs['word_confidences'] = word_confidences

        return inputs, outputs


    """
    Build graph for training
    """
    def build_train(self):

        # this line of code is just a message to inform that batch size should be set to 1 only
        batch_size = 1

        # 以字典的形式保存输入和输出的shape
        inputs = {}
        outputs = {}

        #******************** Define Proposal Module ******************#

        ## dim1: batch, dim2: video sequence length, dim3: video feature dimension
        ## video feature sequence
        #---------------------------------------定义输入的占位符------------------------------------#
        # forward video feature sequence
        video_feat_fw = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat_fw')  # (B,T,C)
        inputs['video_feat_fw'] = video_feat_fw

        # backward video feature sequence
        video_feat_bw = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat_bw')  # (B,T,C)
        inputs['video_feat_bw'] = video_feat_bw

        ## proposal data, densely annotated, in forward direction
        proposal_fw = tf.placeholder(tf.int32, [None, None, self.options['num_anchors']], name='proposal_fw')  # (B,T,120)
        inputs['proposal_fw'] = proposal_fw   # 前向 GT

        ## proposal data, densely annotated, in backward direction
        proposal_bw = tf.placeholder(tf.int32, [None, None, self.options['num_anchors']], name='proposal_bw')  # (B,T,120)
        inputs['proposal_bw'] = proposal_bw  # 反向 GT

        # 在进行Caption时需要选取前N个proposal进行Caption,保存用于做caption的提议，这个值是什么样子的？？
        ## proposal to feed into captioning module, i choose high tiou proposals for training captioning module, forward pass
        proposal_caption_fw = tf.placeholder(tf.int32, [None, None], name='proposal_caption_fw')  # (B,T)
        inputs['proposal_caption_fw'] = proposal_caption_fw  # 前向 Caption GT

        ## proposal to feed into captioning module, i choose high tiou proposals for training captioning module, backward pass
        proposal_caption_bw = tf.placeholder(tf.int32, [None, None], name='proposal_caption_bw')  # (B,T)
        inputs['proposal_caption_bw'] = proposal_caption_bw  # 反向 Caption GT

        ## weighting for positive/negative labels (solve imbalance data problem)
        proposal_weight = tf.placeholder(tf.float32, [self.options['num_anchors'], 2], name='proposal_weight')  # (120,2)
        inputs['proposal_weight'] = proposal_weight
       
        # ----------------------------------------用于编码输入特征的RNN单元---------------------------------------------#
        rnn_cell_video_fw = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],      # 512
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_video_bw = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer() # 生成正交矩阵的初始化器.
        )

        if self.options['rnn_drop'] > 0:        # 0.3
            print('using dropout in rnn!')
            
        rnn_drop = tf.placeholder(tf.float32)
        inputs['rnn_drop'] = rnn_drop
        
        # 控制RNN的dropout比例
        rnn_cell_video_fw = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_video_fw,
            input_keep_prob=1.0 - rnn_drop,   # keep_ratio = 0.7
            output_keep_prob=1.0 - rnn_drop 
        )
        rnn_cell_video_bw = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_video_bw,
            input_keep_prob=1.0 - rnn_drop,
            output_keep_prob=1.0 - rnn_drop 
        )
        
        
        with tf.variable_scope('proposal_module') as proposal_scope:

            '''video feature sequence encoding: forward pass
            '''
            #---------------------------------用RNN对输入特征进行编码，用全连接层预测每个提议的得分------------------------------------#
            with tf.variable_scope('video_encoder_fw') as scope:
                #sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
                sequence_length = tf.expand_dims(tf.shape(video_feat_fw)[1], axis=0)   # (1,)
                initial_state = rnn_cell_video_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
                
                #(1,?,512)
                # tf.nn.dynamic_rnn 函数是tensorflow封装的用来实现递归神经网络（RNN）的函数
                
                # 利用RNN对输入视频特征进行编码
                rnn_outputs_fw, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell_video_fw,   # cell：LSTM、GRU等的记忆单元。cell参数代表一个LSTM或GRU的记忆单元，也就是一个cell。
                    inputs=video_feat_fw,  
                    sequence_length=sequence_length,   # 输入序列的最长值
                    initial_state=initial_state,
                    dtype=tf.float32
                )
            # (1,T,512)-->(T,512)
            rnn_outputs_fw_reshape = tf.reshape(rnn_outputs_fw, [-1, self.options['rnn_size']], name='rnn_outputs_fw_reshape')  # (T,512)

            # 只需要用全连接层输出每个anchor的置信度得分就可以了，提议是提前就预设好的
            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal_fw') as scope:
                logit_output_fw = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_fw_reshape, 
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )   # (T,512)-->(T,120) ,预测每个anchor属于proposal的得分

            '''video feature sequence encoding: backward pass
            '''
            with tf.variable_scope('video_encoder_bw') as scope:
                #sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
                sequence_length = tf.expand_dims(tf.shape(video_feat_bw)[1], axis=0)
                initial_state = rnn_cell_video_bw.zero_state(batch_size=batch_size, dtype=tf.float32)
                
                rnn_outputs_bw, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell_video_bw, 
                    inputs=video_feat_bw, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
            
             # (1,T,512)-->(T,512)
            rnn_outputs_bw_reshape = tf.reshape(rnn_outputs_bw, [-1, self.options['rnn_size']], name='rnn_outputs_bw_reshape')

            # 只需要用全连接层输出每个anchor的置信度得分就可以了，提议是提前就预设好的
            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal_bw') as scope:
                logit_output_bw = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_bw_reshape, 
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )

           #---------------------------------------------------------------------------------------------#
        
        # calculate multi-label loss: use weighted binary cross entropy objective
        proposal_fw_reshape = tf.reshape(proposal_fw, [-1, self.options['num_anchors']], name='proposal_fw_reshape')
        proposal_fw_float = tf.to_float(proposal_fw_reshape)  # (T,120)
        proposal_bw_reshape = tf.reshape(proposal_bw, [-1, self.options['num_anchors']], name='proposal_bw_reshape')
        proposal_bw_float = tf.to_float(proposal_bw_reshape)  # (T,120)

        # weighting positive samples
        weight0 = tf.reshape(proposal_weight[:, 0], [-1, self.options['num_anchors']])
        
        # weighting negative samples
        weight1 = tf.reshape(proposal_weight[:, 1], [-1, self.options['num_anchors']])

        # tile weight batch_size times
        weight0 = tf.tile(weight0, [tf.shape(logit_output_fw)[0], 1])    # (T,120)
        weight1 = tf.tile(weight1, [tf.shape(logit_output_fw)[0], 1])    # (T,120)

        # get weighted sigmoid xentropy loss
        loss_term_fw = tf.nn.weighted_cross_entropy_with_logits(targets=proposal_fw_float, logits=logit_output_fw, pos_weight=weight0)
        loss_term_bw = tf.nn.weighted_cross_entropy_with_logits(targets=proposal_bw_float, logits=logit_output_bw, pos_weight=weight0)

        # 先对每个时序位置的120个anchor的Loss求和
        loss_term_fw_sum = tf.reduce_sum(loss_term_fw, axis=-1, name='loss_term_fw_sum')
        loss_term_bw_sum = tf.reduce_sum(loss_term_bw, axis=-1, name='loss_term_bw_sum')

        # 再对所有的时序位置求和除以所有的anchor数
        proposal_fw_loss = tf.reduce_sum(loss_term_fw_sum) / (float(self.options['num_anchors'])*tf.to_float(tf.shape(video_feat_fw)[1]))
        proposal_bw_loss = tf.reduce_sum(loss_term_bw_sum) / (float(self.options['num_anchors'])*tf.to_float(tf.shape(video_feat_bw)[1]))
        proposal_loss = (proposal_fw_loss + proposal_bw_loss) / 2.

        # summary data, for visualization using Tensorboard
        tf.summary.scalar('proposal_fw_loss', proposal_fw_loss)   
        tf.summary.scalar('proposal_bw_loss', proposal_bw_loss)   
        tf.summary.scalar('proposal_loss', proposal_loss)         

        # outputs from proposal module
        outputs['proposal_fw_loss'] = proposal_fw_loss
        outputs['proposal_bw_loss'] = proposal_bw_loss
        outputs['proposal_loss'] = proposal_loss


        #*************** Define Captioning Module *****************#

        ## caption data: densely annotate sentences for each time step of a video, use mask data to mask out time steps when no caption should be output
         # caption_sep_len = 30
        # 存放真实的caption
        caption = tf.placeholder(tf.int32, [None, None, self.options['caption_seq_len']], name='caption')  # (B,T,30)  
        caption_mask = tf.placeholder(tf.int32, [None, None, self.options['caption_seq_len']], name='caption_mask')  # (B,T,30)
        inputs['caption'] = caption
        inputs['caption_mask'] = caption_mask

        # proposal_caption_fw种记录了哪个时序位置有满足用于caption的提议
        proposal_caption_fw_reshape = tf.reshape(proposal_caption_fw, [-1], name='proposal_caption_fw_reshape')  # (B,T) ---> (T)  
        proposal_caption_bw_reshape = tf.reshape(proposal_caption_bw, [-1], name='proposal_caption_bw_reshape')  # (B,T) ---> (T)  

        # use correct or 'nearly correct' proposal output as input to the captioning module
        
        # boolean_mask中保存着满足条件的时序位置，前向和反向的boolean——mask是一样的，只是其对应的时序位置不同
        boolean_mask = tf.greater(proposal_caption_fw_reshape, 0, name='boolean_mask')  # 选取满足条件的时序位置

        # guarantee that at least one pos has True value
        """
        z = tf.multiply(a, b)
        result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
        """
        boolean_mask = tf.cond(tf.equal(tf.reduce_sum(tf.to_int32(boolean_mask)), 0), lambda: tf.concat([boolean_mask[:-1], tf.constant([True])], axis=-1), lambda: boolean_mask)

        # select input video state
        feat_len = tf.shape(video_feat_fw)[1]   # T 
        
        # 通过boolean_mask函数选取对应位置，由于正向时，时序位置和boolean_mask的位置是对应的
        forward_indices = tf.boolean_mask(tf.range(feat_len), boolean_mask)  # tf.boolean_mask直接返回对应mask位置的值，返回的是tensor值
        
        # rnn_outputs_fw_reshape : (N,512)
        event_feats_fw = tf.boolean_mask(rnn_outputs_fw_reshape, boolean_mask)  
        # proposal_caption_bw_reshape : (T)
      
        # 反向时，时序位置和boolean_mask中的位置不是对应的，因此需要用boolean_mask选取出proposal_caption_bw_reshape对应位置的值
        backward_indices = tf.boolean_mask(proposal_caption_bw_reshape, boolean_mask)
        """
        tf.gather:
            类似于数组的索引，可以把向量中某些索引值提取出来，得到新的向量，适用于要提取的索引为不连续的情况。这个函数似乎只适合在一维的情况下使用。
        tf.gather_nd:
            同上，但允许在多维上进行索引
        """
        # rnn_outputs_bw_reshape : (T,512)
        # 获取反向时对应位置的特征
        # backword_indices : (T)--> (T,1)
        # (T,512),(T,1)
        event_feats_bw = tf.gather_nd(rnn_outputs_bw_reshape, tf.expand_dims(backward_indices, axis=-1))
        
        # 开始和结束时间列表
        start_ids = feat_len - 1 - backward_indices
        end_ids = forward_indices
        
        # max_proposal_len = 110   提议对应的特征 (N,110,500)
        event_c3d_seq, _ = self.get_c3d_seq(video_feat_fw[0], start_ids, end_ids, self.options['max_proposal_len']) 
        
        
        context_feats_fw = tf.gather_nd(rnn_outputs_fw_reshape, tf.expand_dims(start_ids, axis=-1))  # (T,512),(T,1) 
        context_feats_bw = tf.gather_nd(rnn_outputs_bw_reshape, tf.expand_dims(feat_len-1-end_ids, axis=-1))

        # proposal feature sequences
        proposal_feats = event_c3d_seq

        # corresponding caption ground truth (batch size  = 1)
        caption_proposed = tf.boolean_mask(caption[0], boolean_mask, name='caption_proposed')  # (N,30)
        caption_mask_proposed = tf.boolean_mask(caption_mask[0], boolean_mask, name='caption_mask_proposed')  # (N,30)

        # the number of proposal-caption pairs for training
        n_proposals = tf.shape(caption_proposed)[0]   # 满足条件的提议个数

        # -------------------------------------用于输出caption的RNN单元-----------------------------#
        rnn_cell_caption = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],  # 隐藏层的神经元个数
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )
        
        rnn_cell_caption = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_caption,
            input_keep_prob=1.0 - rnn_drop,
            output_keep_prob=1.0 - rnn_drop 
        )

        # state_is_tuple:如果为True，则接受和返回的状态是c_state和m_state的2-tuple；如果为False，则他们沿着列轴连接。后一种即将被弃用。
        def get_rnn_cell():
            return tf.contrib.rnn.LSTMCell(num_units=self.options['rnn_size'], state_is_tuple=True, initializer=tf.orthogonal_initializer())

        # multi-layer LSTM
        # num_rnn_layers = 2，构建多隐层神经网络
        # state_is_tuple：true，状态Ct和ht就是分开记录，放在一个tuple中，接受和返回的states是n-tuples，其中n=len(cells)，False，states是concatenated沿着列轴.后者即将弃用。
        multi_rnn_cell_caption = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(self.options['num_rnn_layers'])], state_is_tuple=True)

        caption_loss = 0
        with tf.variable_scope('caption_module') as caption_scope:

            batch_size = n_proposals  # 在训练caption model时batch_size是所有满足条件的提议

            # initialize memory cell and hidden output, note that the returned state is a tuple containing all states for each cell in MultiRNNCell
            state = multi_rnn_cell_caption.zero_state(batch_size=batch_size, dtype=tf.float32)

            # (N,110,512)--> (Nx110,512)
            proposal_feats_reshape = tf.reshape(proposal_feats, [-1, self.options['video_feat_dim']], name='proposal_feats_reshape')

            # event_feats_fw : (N,512)
            # event_feats_bw : (N,512)
            event_hidden_feats = tf.concat([event_feats_fw, event_feats_bw], axis=-1)   # (N,1024) 

            # (N,1024)----->(N,110x1024)
            event_hidden_feats_tile = tf.tile(event_hidden_feats, [1, self.options['max_proposal_len']])  # (N,110x1024)  112640 = 110x1024
            
            # (Nx110,1024)
            event_hidden_feats_reshape = tf.reshape(event_hidden_feats_tile, [-1, 2*self.options['rnn_size']]) # (Nx110,1024)


            ''' 
            The caption data should be prepared in equal length, namely, with length of 'caption_seq_len'
            ## use caption mask data to mask out loss from sequence after end of token (<END>)
            Only the first loop create variable, the other loops reuse them
            '''
            for i in range(self.options['caption_seq_len']-1):

                if i > 0:
                    caption_scope.reuse_variables()

                # word embedding
                word_embed = self.build_caption_embedding(caption_proposed[:, i])   # caption_proposed : (N,30)，word_embed : (N,512)

                # calculate attention over proposal feature elements
                # state[:, 1] return all hidden states for all cells in MultiRNNCell
                h_state = tf.concat([s[1] for s in state], axis=-1)  # (N,1024)
                h_state_tile = tf.tile(h_state, [1, self.options['max_proposal_len']])  # (N,110x1024) 
                h_state_reshape = tf.reshape(h_state_tile, [-1, self.options['num_rnn_layers']*self.options['rnn_size']]) # (Nx110,1024)
                
                # proposal_feats_reshape : (Nx110,500)       # 视频特征
                # h_state_reshape ： (Nx110,1024)     # 隐状态
                # event_hidden_feats_reshape : (Nx110,1024)  # 上下文特征
                
                # (Nx110,2548)
                feat_state_concat = tf.concat([proposal_feats_reshape, h_state_reshape, event_hidden_feats_reshape], axis=-1, name='feat_state_concat')
                #feat_state_concat = tf.concat([tf.reshape(tf.tile(word_embed, [1, self.options['max_proposal_len']]), [-1, self.options['word_embed_size']]), proposal_feats_reshape, h_state_reshape, event_hidden_feats_reshape], axis=-1, name='feat_state_concat')

                #-------------------------------------------对提议特征进行attention--------------------------------#
                
                # use a two-layer network to model attention over video feature sequence when predicting next word (dynamic)
                # 注意力权重的计算相当于综合考虑了视频特征，上下文向量，cell隐状态
                # (Nx110,2548)-->(Nx110,512)-->(Nx110,1)
                with tf.variable_scope('attention') as attention_scope:
                    attention_layer1 = tf.contrib.layers.fully_connected(
                        inputs = feat_state_concat,
                        num_outputs = self.options['attention_hidden_size'],  # 512
                        activation_fn = tf.nn.tanh,
                        weights_initializer = tf.contrib.layers.xavier_initializer()
                    )  # (Nx110,512)
                    attention_layer2 = tf.contrib.layers.fully_connected(
                        inputs = attention_layer1,
                        num_outputs = 1,
                        activation_fn = None,
                        weights_initializer = tf.contrib.layers.xavier_initializer()
                    )  # (Nx110,1)

                # reshape to match
                attention_reshape = tf.reshape(attention_layer2, [-1, self.options['max_proposal_len']], name='attention_reshape')  # (N,110)
                attention_score = tf.nn.softmax(attention_reshape, dim=-1, name='attention_score')  # (N,110)
                attention = tf.reshape(attention_score, [-1, 1, self.options['max_proposal_len']], name='attention') # (N,1,110)

                # attended video feature
                # (N,1,110)x(N,110,500) = (N, 1, 500)
                attended_proposal_feat = tf.matmul(attention, proposal_feats, name='attended_proposal_feat')
                # (N,500)
                attended_proposal_feat_reshape = tf.reshape(attended_proposal_feat, [-1, self.options['video_feat_dim']], name='attended_proposal_feat_reshape')

                # ----------------------------------上下文门机制-------------------------------------#
                if self.options['no_context']:
                    proposal_feats_full = attended_proposal_feat_reshape
                else:
                    if self.options['context_gating']:
                        # model a gate to weight each element of context and feature
                        attended_proposal_feat_reshape = tf.nn.tanh(attended_proposal_feat_reshape)  # (N,500)
                        with tf.variable_scope('context_gating'):
                            '''
                            context_feats_transform = tf.contrib.layers.fully_connected(
                                inputs=event_hidden_feats,
                                num_outputs=self.options['video_feat_dim'],
                                activation_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer()
                            )
                            '''

                            context_feats_transform = event_hidden_feats   # (N,1024)

                            # 将attention后的特征投影到同一空间(1024维)
                            proposal_feats_transform = tf.contrib.layers.fully_connected(
                                inputs = attended_proposal_feat_reshape,
                                num_outputs = 2*self.options['rnn_size'],  # 1024
                                activation_fn = tf.nn.tanh,
                                weights_initializer = tf.contrib.layers.xavier_initializer()
                            )
                            
                            # context gating，
                            # gate ： 控制上下文特征的权重
                            # 1-gate ： 提议特征对应的权重
                            
                             # word_embed : (N,512)   # 词嵌入特征
                             # h_state : (N,1024)     # hidden state
                             # context_feats_transform : (N,1024)  # 对应经lstm编码后整个视频的状态也就是上下文信息
                             # proposal_feats_transform : (N,1024) # 提议原始c3d特征
                                
                             # (N,3584)--> (N,1024)
                            gate = tf.contrib.layers.fully_connected(
                     
                                inputs=tf.concat([word_embed, h_state, context_feats_transform, proposal_feats_transform], axis=-1),
                                num_outputs=2*self.options['rnn_size'],
                                activation_fn=tf.nn.sigmoid,
                                weights_initializer=tf.contrib.layers.xavier_initializer()
                            )
                            
                            # (N,1024)x(N,1024) = (N,1024),相当于对每个通道分配不同的权重
                            gated_context_feats = tf.multiply(context_feats_transform, gate)
                            gated_proposal_feats = tf.multiply(proposal_feats_transform, 1.-gate)  # 
                            proposal_feats_full = tf.concat([gated_context_feats, gated_proposal_feats], axis=-1)  # 生成最终用于caption的特征
                            
                    else:
                        proposal_feats_full = tf.concat([event_hidden_feats, attended_proposal_feat_reshape], axis=-1)


                # proposal feature embedded into word space
                # 将提议特征用全连接层进行降维 (N,1024)--->(N,512)
                proposal_feat_embed = self.build_video_feat_embedding(proposal_feats_full)

                # proposal_feat_embed : (N,512)
                # word_embed : (N,512)
                # caption_ouput : (N,512)
                # state : 
                #    (N,512)
                #    (N,512)
                caption_output, state = multi_rnn_cell_caption(tf.concat([proposal_feat_embed, word_embed], axis=-1), state)

                # predict next word  (N,512)---> (N,4414)
                with tf.variable_scope('logits') as logits_scope:
                    logits = tf.contrib.layers.fully_connected(
                        inputs=caption_output,
                        num_outputs=self.options['vocab_size'],
                        activation_fn=None
                    )

               # i+1 用于获取真实label
               # 因为是通过i来预测i+1位置的单词
                labels = caption_proposed[:, i+1] # predict next word

                # loss term
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                output_mask = tf.to_float(caption_mask_proposed[:,i])  # caption_mask_proposed用于指示单词结束的地方，因为句子长度不一样
                loss = tf.reduce_sum(tf.multiply(loss, output_mask))
                
                caption_loss = caption_loss + loss

        # mean loss for each word
        caption_loss = caption_loss / (tf.to_float(batch_size)*tf.to_float(tf.reduce_sum(caption_mask_proposed)) + 1)

        tf.summary.scalar('caption_loss', caption_loss)
        # tf.add_n: 函数是实现一个列表的元素的相加。就是输入的对象是一个列表，列表里的元素可以是向量，矩阵等但没有广播功能
        # tf.nn.l2_loss : 利用L2范数来计算张量的误差值
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not v.name.startswith('caption_module/word_embed')])
        # weight_proposal = 1
        # weight_caption = 5
        total_loss = self.options['weight_proposal']*proposal_loss + self.options['weight_caption']*caption_loss
        tf.summary.scalar('total_loss', total_loss)

        outputs['caption_loss'] = caption_loss
        outputs['loss'] = total_loss
        outputs['reg_loss'] = reg_loss
        outputs['n_proposals'] = n_proposals

        return inputs, outputs

    # 这里面是经典的tensorflow中的循环的写法，可以借鉴一下
    """get c3d proposal representation (feature sequence), given start end feature ids
       video_feat_sequence : (B,T,C)
       # start_ids 和 end_ids中包含有N个满足条件的提议的起止时间
       start_ids : 开始时间列表
       end_ids : 结束时间列表
       max_clip_len : 110 (对于不到110的，padding到110)
    """
   
    def get_c3d_seq(self, video_feat_sequence, start_ids, end_ids, max_clip_len):
        # 对于时长为T的视频，可能有s个满足条件的时间点
        ind = tf.constant(0)   
        N = tf.shape(start_ids)[0]  # 满足条件的时间点个数
        event_c3d_sequence = tf.fill([0, max_clip_len, self.options['video_feat_dim']], 0.)    # (0,110,500)
        event_c3d_mask = tf.fill([0, max_clip_len], 0.)   # (0,110)
        event_c3d_mask = tf.to_int32(event_c3d_mask)

        def condition(ind, event_c3d_sequence, event_c3d_mask):
            return tf.less(ind, N)

        def body(ind, event_c3d_sequence, event_c3d_mask):
            start_id = start_ids[ind]
            end_id = end_ids[ind]
            c3d_feats =video_feat_sequence[start_id:end_id]
            # padding if needed
            clip_len = end_id - start_id
             # 获取提议C3D特征
             # 将所有提议的特征填充到max_clip_len = 110
            c3d_feats = tf.cond(tf.less(clip_len, max_clip_len), lambda: tf.concat([c3d_feats, tf.fill([max_clip_len-clip_len, self.options['video_feat_dim']], 0.)], axis=0), lambda: c3d_feats[:max_clip_len])
            c3d_feats = tf.expand_dims(c3d_feats, axis=0)  # (1,110,500)
            event_c3d_sequence = tf.concat([event_c3d_sequence, c3d_feats], axis=0)

            # ----------------------------------------获取提议对应的时长mask,没有用-----------------------------------------#
            this_mask = tf.cond(tf.less(clip_len, max_clip_len), lambda: tf.concat([tf.fill([clip_len], 1.), tf.fill([max_clip_len-clip_len], 0.)], axis=0), lambda: tf.fill([max_clip_len], 1.))
            this_mask = tf.expand_dims(this_mask, axis=0)  # (1,110)
            this_mask = tf.to_int32(this_mask)  # ()
            event_c3d_mask = tf.concat([event_c3d_mask, this_mask], axis=0)
            

            return tf.add(ind, 1), event_c3d_sequence, event_c3d_mask

        _, event_c3d_sequence, event_c3d_mask = tf.while_loop(condition, body, loop_vars=[ind, event_c3d_sequence, event_c3d_mask], shape_invariants=[ind.get_shape(), tf.TensorShape([None, None, self.options['video_feat_dim']]), tf.TensorShape([None, None])])


        return event_c3d_sequence, event_c3d_mask   # (N,110,500),(N,110)

    
