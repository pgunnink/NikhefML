from keras.layers import Input, Reshape, Conv2D, concatenate, SeparableConv2D, Flatten,\
    Dense
from keras import Model
from keras.optimizers import Adam
from custom_metrics import metric_degrees_difference


def baseModelDense(N_stations=1,
                   features=2,
                   length_trace=80,
                   trace_filter_1=64,
                   filter_size_1=7,
                   stride_1=4,
                   trace_filter_2=32,
                   filter_size_2=7,
                   stride_2=4,
                   trace_filter_3=10,
                   filter_size_3=4,
                   stride_3=4,):
    '''
    The basic model as described by https://arxiv.org/pdf/1708.00647.pdf, but flattened
    out
    '''
    input_traces = Input(shape=(N_stations * 4, length_trace), dtype='float32',
                         name='trace_input')
    reshape_traces = Reshape((N_stations * 4, length_trace, 1))(input_traces)
    input_metadata = Input(shape=(N_stations * 4, features), dtype='float32',
                           name='metadata_input')

    process_metadata = Reshape((N_stations, 4, features))(input_metadata)

    Trace = Conv2D(trace_filter_1, (1, filter_size_1), strides=(1, stride_1), padding='valid',
                   activation='relu', data_format='channels_last',
                   kernel_initializer='he_normal', name='first_trace_conv')(
        reshape_traces)
    Trace = Conv2D(trace_filter_2, (1, filter_size_2), strides=(1, stride_2),
                   padding='valid',
                   activation='relu', kernel_initializer='he_normal',
                   name='second_trace_conv' )(Trace)
    Trace = Conv2D(trace_filter_3, (1, filter_size_3), strides=(1, stride_3),
                   padding='valid',
                   activation='relu',
                   kernel_initializer='he_normal', name='third_trace_conv')(Trace)
    TraceResult = Reshape((N_stations, 4, -1))(Trace)

    x = concatenate([TraceResult, process_metadata])

    x = Flatten()(x)
    x = Dense(100, kernel_initializer='he_normal', activation='relu')(x)
    x = Dense(50, kernel_initializer='he_normal', activation='relu')(x)
    x = Dense(50, kernel_initializer='he_normal', activation='relu')(x)

    Output = Dense(3)(x)

    model = Model(inputs=[input_traces, input_metadata], outputs=Output)

    A = Adam(lr=0.001)
    model.compile(optimizer=A, loss='mse', metrics=[metric_degrees_difference])

    return model