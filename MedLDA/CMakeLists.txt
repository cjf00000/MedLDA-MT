#add_library(ldalibs stochastic_count.cpp corpus.cpp utils.cpp
#                    flags.cpp base_lda.cpp cvsEM.cpp sgd.cpp gd.cpp
#                    goem.cpp bestscvb0.cpp bestscvb1.cpp mfvoem.cpp
#                    cvb0oem.cpp cvboem.cpp icvboem.cpp ncvboem.cpp
#                    rsgd.cpp smd.cpp svi.cpp)
#target_link_libraries(ldalibs gflags)

#set(LINK_LIBS ldalibs gflags)

add_executable(svm svm_main.cpp svm.cpp corpus.cpp)
target_link_libraries(svm gflags)

add_executable(medlda medlda_main.cpp corpus.cpp medlda.cpp svm.cpp utils.cpp sparse_vector.cpp bit.cpp)
target_link_libraries(medlda gflags)

add_executable(bit_test bit.cpp bit_test.cpp)

#add_executable(lda main.cpp)
#target_link_libraries(lda ${LINK_LIBS})
