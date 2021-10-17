StepVerifier.create(result).expectErrorSatisfies((exception) -> {
  if (!(exception instanceof SSLException)) {
    assertThat(exception).hasCauseInstanceOf(SSLException.class);
  }
}).verify(Duration.ofSeconds(10));