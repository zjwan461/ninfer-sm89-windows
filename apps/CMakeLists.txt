add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

# Fork-local server launchers. The fork ships one launcher per compile-time SM profile
# (start-bonsai-rtx4090.bat / start-qwen38-rtx4090.bat for 128 SMs, the -rtx4080s pair for 80), so
# stage the pair matching this build's NINFER_TARGET_SM_COUNT next to the server binary: the
# launchers resolve `%~dp0ninfer-serve.exe`, and their only difference is the prefill chunk the
# matching wave geometry wants. They are fork conveniences that may be absent (e.g. a clean
# upstream checkout), so guard on existence instead of failing configure when they are missing.
if(NINFER_TARGET_SM_COUNT EQUAL 128)
  set(NINFER_LAUNCHER_PROFILE rtx4090)
elseif(NINFER_TARGET_SM_COUNT EQUAL 80)
  set(NINFER_LAUNCHER_PROFILE rtx4080s)
else()
  # 128 and 80 are the two qualified profiles; another legal SM count gets no launcher.
  set(NINFER_LAUNCHER_PROFILE "")
endif()

if(NINFER_LAUNCHER_PROFILE)
  foreach(ninfer_launcher_model IN ITEMS bonsai qwen38)
    set(ninfer_launcher
        "${PROJECT_SOURCE_DIR}/start-${ninfer_launcher_model}-${NINFER_LAUNCHER_PROFILE}.bat")
    if(EXISTS "${ninfer_launcher}")
      add_custom_command(TARGET ninfer-serve POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${ninfer_launcher}"
                "$<TARGET_FILE_DIR:ninfer-serve>"
        COMMENT "Staging ${ninfer_launcher} next to the server binary")
    endif()
  endforeach()
endif()

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
