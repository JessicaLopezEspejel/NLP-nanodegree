from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
from keras.layers import Dropout

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    bn_rnn = input_data
    for i in range(recur_layers):
        layer_name='rnn_'+str(i)
        simp_rnn = LSTM(units, activation='relu', return_sequences=True, implementation=2, name=layer_name)(bn_rnn)
        bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn =  Bidirectional(LSTM(output_dim, return_sequences=True, implementation=2, name='rnn'), 
                               merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model




def final_model(input_dim,
                # CNN parameters
                filters=200,
                kernel_size=11,
                conv_stride=2,
                conv_border_mode='same',
                dilation=1,
                cnn_dropout=0.2,
                # RNN parameters
                gru_units=29,
                gru_layers=10,  
                gru_implementation=2,
                # Fully Connected layer parameters
                fc_units=[80],  
                fc_dropout=0.2,
               ):
    """ Build a deep network for speech
    """

    input_data = Input(name='the_input', shape=(None, input_dim))

    # Convolutional layer
    nn = Conv1D(filters,
                kernel_size,
                strides=conv_stride,
                padding=conv_border_mode,
                dilation_rate=dilation,
                activation='relu')(input_data)

    # Add (in order) Batch Normalization,Dropout and Activation
    nn = BatchNormalization()(nn)
    #nn = Dropout(cnn_dropout)(nn)
    #nn = Activation('relu')(nn)

    # TODO: Add bidirectional recurrent layers
    for i in range(gru_layers):
        layer_number = str(i)
        nn =  GRU(gru_units, return_sequences=True,
                                implementation=gru_implementation,
                                name='gru_'+layer_number)(nn)

        nn = BatchNormalization()(nn)


    # TODO: Add a Fully Connected layers
    fc_layers = len(fc_units)
    for i in range(fc_layers):
        nn = TimeDistributed(Dense(units=fc_units[i]))(nn)
        nn = Dropout(fc_dropout)(nn)
        nn = Activation('relu')(nn)

    nn = TimeDistributed(Dense(units=29))(nn)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(nn)

    # TODO: Specify the model
    model = Model(inputs=input_data, outputs=y_pred)

    # TODO: Specify model.output_length: select custom or Udacity version
    model.output_length = lambda x: multi_cnn_output_length(x,
                                                            kernel_size,
                                                            conv_border_mode, conv_stride,
                                                            cnn_layers=1)

    print(model.summary(line_length=110))
    return model



def multi_cnn_output_length(input_length, filter_size, border_mode, stride,
                            dilation=1, cnn_layers=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
       
    if input_length is None:
        return None
    
    # Stacking several convolution layers only works with 'same' padding in this implementation
    if cnn_layers>1:
        assert border_mode in {'same'}
    else:
        assert border_mode in {'same', 'valid'}
    
    length = input_length
    for i in range(cnn_layers):
    
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        if border_mode == 'same':
            output_length = length
        elif border_mode == 'valid':
            output_length = length - dilated_filter_size + 1
                
        length = (output_length + stride - 1) // stride
        
    return length

