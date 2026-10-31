foreach(TEST_NAME IN ITEMS test_segmenting test_kv_layer test_kv_layer_rw
        test_fh_index_ro test_fh_index_rw test_batch_successor
        test_maximal_covering test_min_max_segmentation
        test_multi_level_segmentor test_epoch_reclamation test_fraction
        test_retrain_stats test_lifecycle test_boundaries)
    set(BLOCK_SIZE 0)
    set(FINGER_SPARSITY 0)
    set(TEST_ARGS "")
    if(TEST_NAME STREQUAL "test_fh_index_ro" OR TEST_NAME STREQUAL "test_fh_index_rw")
        set(BLOCK_SIZE 40)
        set(FINGER_SPARSITY 2)
        set(TEST_ARGS GEN 10000 1000 42)
    elseif(TEST_NAME STREQUAL "test_multi_level_segmentor")
        set(TEST_ARGS GEN 10000)
    elseif(TEST_NAME STREQUAL "test_maximal_covering")
        set(TEST_ARGS 10000)
    endif()
    set(NAME "${TEST_NAME}_lat0_typeNONE_b${BLOCK_SIZE}_f${FINGER_SPARSITY}")
    fh_index_add_program(${NAME} "${TEST_NAME}.cpp")
    if(BLOCK_SIZE)
        target_compile_definitions(${NAME} PRIVATE
            BLOCK_SIZE=${BLOCK_SIZE} FINGER_SPARSITY=${FINGER_SPARSITY})
    endif()
    # Existing tests rely on assert; retain it even in optimized builds.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${NAME} PRIVATE -UNDEBUG)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${NAME} ${TEST_ARGS})
    set_tests_properties(${TEST_NAME} PROPERTIES
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
        ENVIRONMENT "PARLAY_NUM_THREADS=4")
endforeach()
